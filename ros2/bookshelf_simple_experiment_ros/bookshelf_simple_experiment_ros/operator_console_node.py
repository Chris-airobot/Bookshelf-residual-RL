#!/usr/bin/env python3
"""Interactive state-gated console for the complete Bookshelf experiment."""

import json
import os
import queue
import select
import termios
import threading
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


SERVICES = {
    "plan_scan": "/bookshelf_simple/plan_scan",
    "accept_slot": "/bookshelf_simple/accept_slot",
    "plan_loading": "/bookshelf_simple/plan_loading",
    "open": "/bookshelf_simple/open_gripper",
    "close": "/bookshelf_simple/close_gripper",
    "plan_preinsert": "/bookshelf_simple/plan_preinsert",
    "execute": "/bookshelf_simple/execute_preinsert",
    "start_policy": "/bookshelf_simple/start_policy",
    "plan_return": "/bookshelf_simple/plan_return_loading",
    "finish_return": "/bookshelf_simple/finish_return",
}


def log_service_response(logger, success, message):
    if success:
        logger.info(str(message))
    else:
        logger.error(str(message))


class OperatorWorkflow:
    START = "start"
    PLANNING_SCAN = "planning_scan"
    SCAN_PLAN_READY = "scan_trajectory_ready"
    EXECUTING_SCAN = "executing_scan"
    SCAN = "scan"
    SLOT_ACCEPTED = "slot_accepted"
    PLANNING_LOADING = "planning_loading"
    LOADING_PLAN_READY = "loading_trajectory_ready"
    EXECUTING_LOADING = "executing_loading"
    LOADING_HOLD = "loading_hold"
    OPENING_FOR_LOAD = "opening_for_load"
    WAITING_FOR_BOOK = "waiting_for_book"
    CLOSING_BOOK = "closing_book"
    BOOK_HELD = "book_held"
    PLANNING_PREINSERT = "planning_preinsert"
    PREINSERT_PLAN_READY = "preinsert_plan_ready"
    EXECUTING_PREINSERT = "executing_preinsert"
    PREINSERT_READY = "preinsert_ready"
    POLICY_RUNNING = "policy_running"
    PUSH_COMPLETE_WAITING_RETURN = "push_complete_waiting_return"
    PLANNING_RETURN = "planning_return"
    RETURN_PLAN_READY = "return_trajectory_ready"
    EXECUTING_RETURN = "executing_return"
    OPENING_AFTER_RETURN = "opening_after_return"
    RETURN_FAILED_WAITING = "return_failed_waiting"
    READY_FOR_NEXT_BOOK = "ready_for_next_book"

    PLAN_COMMANDS = {
        "g": ("plan_scan", (START, READY_FOR_NEXT_BOOK), PLANNING_SCAN, "scan"),
        "l": ("plan_loading", (SLOT_ACCEPTED,), PLANNING_LOADING, "loading"),
        "p": ("plan_preinsert", (BOOK_HELD,), PLANNING_PREINSERT, "preinsert"),
        "h": (
            "plan_return",
            (PUSH_COMPLETE_WAITING_RETURN,),
            PLANNING_RETURN,
            "return_loading",
        ),
    }
    READY_BY_KIND = {
        "scan": SCAN_PLAN_READY,
        "loading": LOADING_PLAN_READY,
        "preinsert": PREINSERT_PLAN_READY,
        "return_loading": RETURN_PLAN_READY,
    }
    EXECUTING_BY_KIND = {
        "scan": EXECUTING_SCAN,
        "loading": EXECUTING_LOADING,
        "preinsert": EXECUTING_PREINSERT,
        "return_loading": EXECUTING_RETURN,
    }

    def __init__(self):
        self.state = self.START
        self.pending = None
        self.pending_plan_kind = None
        self.plan_request_accepted = False
        self.planning_status_seen = False
        self.plan_ready_seen = False
        self.execution_request_accepted = False
        self.execution_done_seen = False
        self.plan_origin_state = None

    def command(self, key):
        key = str(key).lower()
        if key == "q":
            return "quit", None
        if self.pending is not None:
            return None, f"Waiting for pending {self.pending} response."
        if key in self.PLAN_COMMANDS:
            action, allowed, state, kind = self.PLAN_COMMANDS[key]
            if self.state not in allowed:
                return None, f"{key.upper()} is unavailable in {self.state}."
            self.pending_plan_kind = kind
            self.plan_origin_state = self.state
            self.plan_request_accepted = False
            self.planning_status_seen = False
            self.plan_ready_seen = False
            self.execution_request_accepted = False
            self.execution_done_seen = False
            self.state = state
            self.pending = action
            return action, None
        simple = {
            "s": (self.SCAN, "accept_slot"),
            "o": (self.LOADING_HOLD, "open"),
            "c": (self.WAITING_FOR_BOOK, "close"),
            "i": (self.PREINSERT_READY, "start_policy"),
        }
        if key == "e":
            if self.state not in self.READY_BY_KIND.values() or not self.pending_plan_kind:
                return None, "E requires a valid reviewed trajectory."
            self.pending = "execute"
            return "execute", None
        if key not in simple:
            return None, "Unknown key. Use G, S, L, O, C, P, E, I, H, or Q."
        required, action = simple[key]
        if self.state != required:
            return None, f"{key.upper()} is unavailable in {self.state}."
        self.pending = action
        return action, None

    def service_result(self, action, success):
        if self.pending == action:
            self.pending = None
        if action in ("plan_scan", "plan_loading", "plan_preinsert", "plan_return"):
            self.plan_request_accepted = bool(success)
            if not success:
                fallback = {
                    "plan_scan": self.plan_origin_state or self.START,
                    "plan_loading": self.SLOT_ACCEPTED,
                    "plan_preinsert": self.BOOK_HELD,
                    "plan_return": self.PUSH_COMPLETE_WAITING_RETURN,
                }
                self.state = fallback[action]
                self.pending_plan_kind = None
                self.plan_origin_state = None
            elif self.plan_ready_seen:
                self.state = self.READY_BY_KIND[self.pending_plan_kind]
            return
        if action == "execute":
            self.execution_request_accepted = bool(success)
            if success:
                if self.execution_done_seen:
                    return self._complete_execution(self.pending_plan_kind)
                self.state = self.EXECUTING_BY_KIND[self.pending_plan_kind]
            else:
                self.state = self.READY_BY_KIND[self.pending_plan_kind]
            return
        transitions = {
            "accept_slot": (self.SLOT_ACCEPTED, self.SCAN),
            "open": (self.OPENING_FOR_LOAD, self.LOADING_HOLD),
            "close": (self.CLOSING_BOOK, self.WAITING_FOR_BOOK),
            "start_policy": (self.POLICY_RUNNING, self.PREINSERT_READY),
            "finish_return": (self.OPENING_AFTER_RETURN, self.RETURN_FAILED_WAITING),
        }
        if action in transitions:
            self.state = transitions[action][0 if success else 1]

    def preinsert_status(self, phase, plan_kind=None):
        kind = plan_kind or self.pending_plan_kind
        if phase in (
            "planning",
            "planning_ik_branches",
            "verifying_direct_trajectory",
        ) and kind:
            self.planning_status_seen = True
        elif (
            phase == "awaiting_execute_confirmation"
            and self.planning_status_seen
            and kind in self.READY_BY_KIND
        ):
            self.plan_ready_seen = True
            self.pending_plan_kind = kind
            if self.plan_request_accepted:
                self.state = self.READY_BY_KIND[kind]
        elif phase == "done" and kind:
            self.execution_done_seen = True
            if self.execution_request_accepted:
                return self._complete_execution(kind)
        elif phase in ("failed", "rejected"):
            if kind in self.READY_BY_KIND:
                self.state = self._retry_state(kind)
                self.pending_plan_kind = None
                self.plan_origin_state = None
        return None

    def _retry_state(self, kind):
        if kind == "scan":
            return self.plan_origin_state or self.START
        return {
            "loading": self.SLOT_ACCEPTED,
            "preinsert": self.BOOK_HELD,
            "return_loading": self.PUSH_COMPLETE_WAITING_RETURN,
        }[kind]

    def _complete_execution(self, kind):
        self.execution_request_accepted = False
        self.execution_done_seen = False
        self.pending_plan_kind = None
        self.plan_origin_state = None
        if kind == "scan":
            self.state = self.SCAN
        elif kind == "loading":
            self.state = self.LOADING_HOLD
        elif kind == "preinsert":
            self.state = self.PREINSERT_READY
        else:
            self.state = self.OPENING_AFTER_RETURN
            return "finish_return"
        return None

    def operator_action_status(self, action, success):
        if action == "open":
            self.state = self.WAITING_FOR_BOOK if success else self.LOADING_HOLD
        elif action == "close":
            self.state = self.BOOK_HELD if success else self.WAITING_FOR_BOOK
        elif action in ("return_open", "return_failed") and not success:
            self.state = self.RETURN_FAILED_WAITING
        elif action == "ready" and success:
            self.state = self.READY_FOR_NEXT_BOOK

    def policy_status(self, phase):
        if phase == "episode_complete":
            self.state = self.PUSH_COMPLETE_WAITING_RETURN


class TtyKeyboardReader:
    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run(self):
        descriptor = None
        original = None
        try:
            descriptor = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            original = termios.tcgetattr(descriptor)
            tty.setcbreak(descriptor)
            while not self.stop_event.is_set():
                ready, _, _ = select.select([descriptor], [], [], 0.25)
                if ready:
                    for character in os.read(descriptor, 32).decode(errors="ignore"):
                        if character.strip():
                            self.output_queue.put(("key", character.lower()))
        except Exception as error:
            self.output_queue.put(("error", str(error)))
        finally:
            if descriptor is not None and original is not None:
                termios.tcsetattr(descriptor, termios.TCSADRAIN, original)
            if descriptor is not None:
                os.close(descriptor)


class RealExperimentOperator(Node):
    def __init__(self):
        super().__init__("bookshelf_real_experiment_operator")
        self.workflow = OperatorWorkflow()
        self.events = queue.Queue()
        self.service_clients = {
            action: self.create_client(Trigger, service) for action, service in SERVICES.items()
        }
        self.create_subscription(String, "/bookshelf_simple/status", self._preinsert, 10)
        self.create_subscription(String, "/bookshelf_simple/policy/status", self._policy, 10)
        self.create_subscription(
            String, "/bookshelf_simple/operator_action_status", self._operator_action, 10
        )
        self.keyboard = TtyKeyboardReader(self.events)
        self.keyboard.start()
        self.create_timer(0.05, self._drain_keyboard)
        self._render()

    def _render(self):
        print(f"\nBOOKSHELF EXPERIMENT [{self.workflow.state}]", flush=True)
        print(
            "Keys: G=scan S=freeze L=loading O=open C=close P=plan "
            "E=execute I=policy H=return Q=quit", flush=True,
        )

    def _drain_keyboard(self):
        while not self.events.empty():
            event, value = self.events.get_nowait()
            if event == "error":
                self.get_logger().error(value)
            else:
                self._handle_key(value)

    def _call(self, action):
        client = self.service_clients[action]
        if not client.service_is_ready():
            self.workflow.service_result(action, False)
            self.get_logger().error(f"Service is not ready: {SERVICES[action]}")
            self._render()
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda done, action=action: self._service_response(action, done))

    def _handle_key(self, key):
        action, message = self.workflow.command(key)
        if message:
            self.get_logger().warning(message)
        elif action == "quit":
            self.keyboard.stop()
            rclpy.shutdown()
        else:
            self._call(action)

    def _service_response(self, action, future):
        try:
            response = future.result()
            success, message = bool(response.success), str(response.message)
        except Exception as error:
            success, message = False, str(error)
        followup = self.workflow.service_result(action, success)
        log_service_response(
            self.get_logger(), success,
            f"{action}: {'accepted' if success else 'failed'} - {message}",
        )
        self._render()
        if followup:
            self.workflow.pending = followup
            self._call(followup)

    @staticmethod
    def _decode(message):
        try:
            return json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _preinsert(self, message):
        status = self._decode(message)
        if status is None:
            return
        followup = self.workflow.preinsert_status(
            str(status.get("phase", "")), status.get("plan_kind")
        )
        self._render()
        if followup:
            self.workflow.pending = followup
            self._call(followup)

    def _policy(self, message):
        status = self._decode(message)
        if status:
            self.workflow.policy_status(str(status.get("phase", "")))
            self._render()

    def _operator_action(self, message):
        status = self._decode(message)
        if status:
            self.workflow.operator_action_status(
                str(status.get("action", "")), bool(status.get("success", False))
            )
            self._render()


def main(args=None):
    rclpy.init(args=args)
    node = RealExperimentOperator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.keyboard.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
