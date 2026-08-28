#!/usr/bin/env python3
"""Interactive, non-blocking console for the reviewed real preinsert workflow."""

from __future__ import annotations

import json
import os
import queue
import select
import termios
import threading
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


SERVICES = {
    "accept_slot": "/bookshelf_simple/accept_slot",
    "plan": "/bookshelf_simple/plan_preinsert",
    "execute": "/bookshelf_simple/execute_preinsert",
}


class OperatorWorkflow:
    """Small ROS-independent safety state machine for keyboard commands."""

    SCAN = "scan"
    SLOT_ACCEPTED = "slot_accepted"
    PLANNING = "planning"
    PLAN_READY = "plan_ready"
    EXECUTING = "executing"
    COMPLETE = "complete"

    def __init__(self):
        self.state = self.SCAN
        self.slot_accepted = False
        self.plan_ready = False
        self.pending = None
        self.plan_cycle_active = False
        self.plan_request_accepted = False
        self.planning_status_seen = False
        self.execution_request_accepted = False
        self.execution_status_seen = False

    def command(self, key):
        key = str(key).lower()
        if key == "q":
            return "quit", None
        if key == "r":
            return None, "No safe reset service exists; R is disabled."
        if self.pending is not None:
            return None, f"Waiting for the pending {self.pending} service response."
        if key == "s":
            if self.state != self.SCAN:
                return None, "S is available only while selecting the scan slot."
            self.pending = "accept_slot"
            return "accept_slot", None
        if key == "p":
            if self.state not in (self.SLOT_ACCEPTED, self.PLAN_READY):
                return None, "P requires an accepted slot and loading preparation."
            self.pending = "plan"
            self.plan_ready = False
            self.plan_cycle_active = True
            self.plan_request_accepted = False
            self.planning_status_seen = False
            self.execution_request_accepted = False
            self.execution_status_seen = False
            self.state = self.PLANNING
            return "plan", None
        if key == "e":
            if (
                self.state != self.PLAN_READY
                or not self.plan_ready
                or not self.plan_cycle_active
                or not self.plan_request_accepted
            ):
                return None, "E is locked until a successful reviewed plan is ready."
            self.pending = "execute"
            self.plan_ready = False
            self.execution_request_accepted = False
            self.execution_status_seen = False
            return "execute", None
        return None, "Unknown key. Use S, P, E, or Q."

    def service_result(self, action, success):
        if self.pending == action:
            self.pending = None
        if action == "accept_slot":
            if success:
                self.slot_accepted = True
                self.state = self.SLOT_ACCEPTED
            else:
                self.state = self.SCAN
        elif action == "plan":
            self.plan_request_accepted = bool(success)
            if not success:
                self.plan_cycle_active = False
                self.plan_ready = False
                self.state = self.SLOT_ACCEPTED if self.slot_accepted else self.SCAN
        elif action == "execute":
            self.execution_request_accepted = bool(success)
            if success:
                self.state = self.EXECUTING
            else:
                self.plan_ready = True
                self.state = self.PLAN_READY

    def status(self, phase, slot_frozen=False):
        self.slot_accepted = self.slot_accepted or bool(slot_frozen)
        if phase == "slot_frozen":
            self.slot_accepted = True
            self.state = self.SLOT_ACCEPTED
        elif phase in (
            "attaching_book",
            "requesting_ik",
            "requesting_ik_branches",
            "planning",
            "planning_ik_branches",
        ) and self.plan_cycle_active and self.state == self.PLANNING:
            self.plan_ready = False
            if self.plan_request_accepted:
                self.planning_status_seen = True
            self.state = self.PLANNING
        elif (
            phase == "awaiting_execute_confirmation"
            and self.plan_cycle_active
            and self.plan_request_accepted
            and self.planning_status_seen
            and self.state == self.PLANNING
        ):
            self.plan_ready = True
            self.state = self.PLAN_READY
        elif phase == "executing" and (
            self.pending == "execute" or self.execution_request_accepted
        ):
            self.execution_status_seen = True
            if self.execution_request_accepted:
                self.plan_ready = False
                self.state = self.EXECUTING
        elif (
            phase == "done"
            and self.execution_request_accepted
            and self.execution_status_seen
            and self.state == self.EXECUTING
        ):
            self.plan_ready = False
            self.state = self.COMPLETE
        elif phase in ("failed", "rejected"):
            self.plan_ready = False
            self.state = self.SLOT_ACCEPTED if self.slot_accepted else self.SCAN
        elif phase in ("awaiting_plan_confirmation", "slot_candidate_ready"):
            self.state = self.SLOT_ACCEPTED if self.slot_accepted else self.SCAN


class TtyKeyboardReader:
    """Read individual keys without ever blocking the ROS executor."""

    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=1.0)

    def _run(self):
        descriptor = None
        original = None
        try:
            descriptor = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            original = termios.tcgetattr(descriptor)
            tty.setcbreak(descriptor)
            while not self.stop_event.is_set():
                ready, _, _ = select.select([descriptor], [], [], 0.25)
                if not ready:
                    continue
                data = os.read(descriptor, 32).decode(errors="ignore")
                for character in data:
                    if character.strip():
                        self.output_queue.put(("key", character.lower()))
        except Exception as error:
            self.output_queue.put(("error", f"keyboard unavailable: {error}"))
        finally:
            if descriptor is not None and original is not None:
                termios.tcsetattr(descriptor, termios.TCSADRAIN, original)
            if descriptor is not None:
                os.close(descriptor)


SCREENS = {
    OperatorWorkflow.SCAN: (
        "[SCAN]\nMove robot using RViz.\nPress S when the slot looks correct."
    ),
    OperatorWorkflow.SLOT_ACCEPTED: (
        "[SLOT ACCEPTED]\nFrozen slot saved. Inspect it in RViz.\n"
        "Move robot to the loading posture, place and close the book manually.\n"
        "Press P when ready."
    ),
    OperatorWorkflow.PLANNING: (
        "[PLANNING]\nRunning singularity-aware IK search..."
    ),
    OperatorWorkflow.PLAN_READY: (
        "[PLAN READY]\nInspect the trajectory in RViz.\n"
        "Press E to execute, or P to replan."
    ),
    OperatorWorkflow.EXECUTING: (
        "[EXECUTING]\nPreinsert motion in progress..."
    ),
    OperatorWorkflow.COMPLETE: (
        "[PREINSERT COMPLETE]\nReady for the existing policy stage."
    ),
}


class RealExperimentOperator(Node):
    def __init__(self):
        super().__init__("bookshelf_real_experiment_operator")
        self.workflow = OperatorWorkflow()
        self.events = queue.Queue()
        self.service_clients = {
            action: self.create_client(Trigger, service)
            for action, service in SERVICES.items()
        }
        status_qos = QoSProfile(
            depth=1,
            # Operator authorization is based only on live transitions from
            # this console session, never replayed retained execution status.
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            String, "/bookshelf_simple/status", self._status_callback, status_qos
        )
        self.keyboard = TtyKeyboardReader(self.events)
        self.keyboard.start()
        self.create_timer(0.05, self._drain_keyboard)
        self._last_screen = None
        self._render(force=True)

    def _render(self, force=False):
        if not force and self.workflow.state == self._last_screen:
            return
        self._last_screen = self.workflow.state
        print("\nBOOKSHELF REAL EXPERIMENT\n", flush=True)
        print(SCREENS[self.workflow.state], flush=True)
        print("Keys: S=freeze slot  P=plan  E=execute reviewed plan  Q=quit console", flush=True)

    def _drain_keyboard(self):
        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                return
            if event == "error":
                self.get_logger().error(value)
            else:
                self._handle_key(value)

    def _handle_key(self, key):
        action, message = self.workflow.command(key)
        if message:
            self.get_logger().warning(message)
            return
        if action == "quit":
            print("Operator console exiting; other launched nodes remain running.", flush=True)
            self.keyboard.stop()
            rclpy.shutdown()
            return
        client = self.service_clients[action]
        if not client.service_is_ready():
            self.workflow.service_result(action, False)
            self.get_logger().error(f"Service is not ready: {SERVICES[action]}")
            self._render(force=True)
            return
        self._render()
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda completed, action=action: self._service_response(action, completed)
        )

    def _service_response(self, action, future):
        try:
            response = future.result()
            success = bool(response.success)
            message = str(response.message)
        except Exception as error:
            success = False
            message = str(error)
        self.workflow.service_result(action, success)
        level = self.get_logger().info if success else self.get_logger().error
        level(f"{action}: {'accepted' if success else 'failed'} - {message}")
        self._render(force=not success or action == "accept_slot")

    def _status_callback(self, message):
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warning("Ignored malformed /bookshelf_simple/status message")
            return
        previous = self.workflow.state
        self.workflow.status(
            str(status.get("phase", "")), bool(status.get("slot_frozen", False))
        )
        if status.get("reason") and status.get("phase") in ("failed", "rejected"):
            self.get_logger().error(str(status["reason"]))
        self._render(force=self.workflow.state != previous)

    def close(self):
        self.keyboard.stop()


def main(args=None):
    rclpy.init(args=args)
    node = RealExperimentOperator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
