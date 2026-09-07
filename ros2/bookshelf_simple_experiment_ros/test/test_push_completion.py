"""Offline PUSH termination tests; never create a ROS node or publish to hardware."""

from types import SimpleNamespace
from unittest.mock import Mock
import math

import numpy as np
import pytest

from bookshelf_simple_experiment_ros.push_completion import (
    JulyPushSuccess, PushCompletion, seating_metrics,
)
from bookshelf_simple_experiment_ros.simple_policy_control_node import SimplePolicyControlNode

SIZE = (0.156, 0.034, 0.236)
WIDTH = 0.03756221756339073  # Sept 7 accepted frozen slot, not approved YAML.
# Full recorded T_slot_book at September 7's first July release crossing.
SEPT7 = np.array([
    [0.9994131893490749, 0.020808358784408842, -0.02720825536136604, 0.02071034829063234],
    [-0.020550732518980368, 0.9997416165164671, 0.009714298635286243, 0.000702571323048351],
    [0.027403363808906768, -0.00914944860314114, 0.9995825845032591, 0.004346103586615732],
    [0, 0, 0, 1],
])


def translated(book, x):
    result = book.copy()
    result[0, 3] += x
    return result


def metrics(book):
    return seating_metrics(book, SIZE, 0.2, WIDTH)


def seated_book():
    book = np.eye(4)
    book[:3, 3] = [.068, 0, .004]
    return book


def test_sept7_old_thirty_mm_cannot_satisfy_july_depth():
    release = metrics(SEPT7)
    assert release["book_depth_m"] == pytest.approx(0.10222889185, abs=1e-8)
    assert release["remaining_depth_m"] == pytest.approx(0.04880819557, abs=1e-8)
    old_end = metrics(translated(SEPT7, 0.03))
    assert old_end["rear_to_mouth_m"] == pytest.approx(-0.03080819557, abs=1e-8)
    assert old_end["remaining_depth_m"] == pytest.approx(0.01880819557, abs=1e-8)
    assert not old_end["rear_ok"] and not old_end["front_ok"]
    assert not old_end["success_geometry"]
    deeper = metrics(translated(SEPT7, 0.048809))
    assert deeper["rear_ok"] and deeper["front_ok"]
    # Translation alone doesn't fix July's lateral-corner clearance gate.
    assert not deeper["lateral_ok"]


def test_four_unique_success_samples_required_and_bad_sample_resets():
    good = metrics(seated_book())
    bad = metrics(translated(SEPT7, 0.03))
    monitor = PushCompletion(.1, 90, .25)
    assert not monitor.observe(good, 1_000_000_000, 1_010_000_000)
    for _ in range(8):
        assert not monitor.observe(good, 1_000_000_000, 1_020_000_000)
    assert monitor.success_samples == 1
    assert not monitor.observe(bad, 1_030_000_000, 1_040_000_000)
    for i in range(4):
        stamp = 1_050_000_000 + i * 40_000_000
        assert monitor.observe(good, stamp, stamp + 10_000_000) == (i == 3)


@pytest.mark.parametrize("stamp,now", [(0, 1), (1, 300_000_001), (2, 1)])
def test_invalid_marker_time_fails_closed(stamp, now):
    with pytest.raises(ValueError):
        PushCompletion(.1, 90, .25).observe(metrics(SEPT7), stamp, now)


def test_regressed_timestamp_and_overshoot_fail_closed():
    monitor = PushCompletion(.1, 90, .25)
    monitor.observe(metrics(SEPT7), 100, 100)
    with pytest.raises(ValueError, match="regressed"):
        monitor.observe(metrics(SEPT7), 99, 100)
    with pytest.raises(ValueError, match="passed"):
        monitor.observe(metrics(translated(SEPT7, .07)), 101, 101)


@pytest.mark.parametrize("component,value", [(1, .02), (2, .02)])
def test_depth_alone_is_not_success(component, value):
    book = translated(SEPT7, .05)
    book[component, 3] = value
    assert not metrics(book)["success_geometry"]


def test_rotated_corner_extent_and_upright_are_checked():
    from bookshelf_simple_experiment_ros.policy_tool_math import make_transform
    yaw = math.radians(9)
    book = make_transform([.07, 0, 0], [0, 0, math.sin(yaw/2), math.cos(yaw/2)])
    assert not metrics(book)["yaw_ok"]
    roll = math.radians(35)
    book = make_transform([.07, 0, 0], [math.sin(roll/2), 0, 0, math.cos(roll/2)])
    assert not metrics(book)["upright_ok"]
    assert not metrics(book)["lateral_ok"]


def harness(book=SEPT7, progress=.052, elapsed=10.):
    now = 20_000_000_000
    current = np.eye(4)
    current[0, 3] = progress
    # TCP originally 21.772 mm behind the modeled release near face.
    origin = translated(SEPT7, .08258037612)
    params = {
        "live_state_max_age_s": .5, "push_book_frame": "target_book_center",
        "contact_tolerance_m": .001, "push_x_uncertainty_m": .005,
        "policy_command_duration_s": .2, "maximum_linear_speed_m_s": .025,
        "maximum_angular_speed_rad_s": .1, "translation_tolerance_m": .0005,
        "rotation_tolerance_rad": .004363323,
        "push_recovery_grace_s": 1.0, "push_recovery_fresh_samples": 3,
    }
    h = SimpleNamespace(
        phase="push", phase_start_ns=now-int(elapsed*1e9), _now_ns=lambda:now,
        push_completion=PushCompletion(.1, 90., .25), push_tcp_path_m=0.,
        push_tcp_progress_m=0., push_start_tcp_xyz=np.zeros(3),
        push_previous_tcp_xyz=np.zeros(3), push_start_xyz=np.zeros(3),
        push_measurement_stamp_ns=None, push_metrics=None,
        push_wait_start_ns=None, push_recovery_last_stamp_ns=None,
        push_recovery_fresh_samples=0, push_wait_reason=None,
        latest_servo_status_ns=now, latest_servo_status=0, observed_servo_statuses=set(),
        _live_input_error=lambda:None, eef_frame="eef", tcp_frame="tcp",
        retreat_direction=np.array([-1.,0.,0.]),
        push_book_origin=origin, push_book_transform=origin.copy(),
        released_book_transform=origin.copy(),
        push_contact_distance_m=None, push_geometric_contact_distance_m=None,
        book_push_distance_m=0., target_eef=translated(current,.003),
        geometry=SimpleNamespace(transform_base_slot=np.eye(4), book_size=SIZE,
                                 slot_depth_m=.2, slot_width_m=WIDTH),
        get_parameter=lambda n:SimpleNamespace(value=params[n]),
        _publish_twist=Mock(), _publish_post_visualization=Mock(),
        _log_phase_event=Mock(), _complete_episode=Mock(), _fail=Mock(),
        twist_publisher=object(),
        _publish_status=Mock(), get_logger=lambda:SimpleNamespace(
            error=Mock(), warning=Mock(), info=Mock()
        ),
    )
    h._push_fresh_transform=Mock(side_effect=lambda frame, age:(
        book.copy() if frame=="target_book_center" else current.copy(), now-10_000_000))
    h._halt_and_fail=lambda reason,**kw:SimplePolicyControlNode._halt_and_fail(h,reason,**kw)
    h._post_servo_status_is_fatal=lambda:SimplePolicyControlNode._post_servo_status_is_fatal(h)
    h._push_diagnostics=lambda:SimplePolicyControlNode._push_diagnostics(h)
    h._begin_push_fresh_state_wait=lambda error:(
        SimplePolicyControlNode._begin_push_fresh_state_wait(h,error)
    )
    h._push_waiting_for_fresh_state_tick=lambda:(
        SimplePolicyControlNode._push_waiting_for_fresh_state_tick(h)
    )
    return h


def test_servo_continues_beyond_old_thirty_mm_and_actor_estimate_is_not_capped():
    h=harness(book=translated(SEPT7,.03), progress=.055)
    SimplePolicyControlNode._push_servo_tick(h)
    assert h.book_push_distance_m > .03
    h._complete_episode.assert_not_called()
    h._fail.assert_not_called()
    assert h._publish_twist.call_args.args[0][0] > 0


def test_servo_success_requires_fresh_measurement_not_fake_push_distance():
    h=harness(book=seated_book(), progress=.075)
    for i in range(4):
        h._now_ns=lambda i=i:20_000_000_000+i*40_000_000
        h._push_fresh_transform=Mock(side_effect=lambda frame, age:(
            seated_book() if frame=="target_book_center" else translated(np.eye(4),.075),
            h._now_ns()-10_000_000))
        SimplePolicyControlNode._push_servo_tick(h)
    h._complete_episode.assert_called_once()
    h._fail.assert_not_called()
    assert h._log_phase_event.call_args.kwargs["completion_reason"]=="SUCCESS_CRITERION"
    np.testing.assert_array_equal(h._publish_twist.call_args.args[0], np.zeros(6))


@pytest.mark.parametrize("progress,elapsed,expected", [(.101,10,"MAX_TRAVEL"),(.05,90,"TIMEOUT")])
def test_servo_budget_stops_without_declaring_success(progress,elapsed,expected):
    h=harness(progress=progress,elapsed=elapsed)
    SimplePolicyControlNode._push_servo_tick(h)
    h._fail.assert_called_once()
    h._complete_episode.assert_not_called()
    assert h._log_phase_event.call_args.kwargs["completion_reason"]==expected
    np.testing.assert_array_equal(h._publish_twist.call_args.args[0], np.zeros(6))
    if expected=="TIMEOUT":h._push_fresh_transform.assert_not_called()


@pytest.mark.parametrize("fault", ["tf", "joints", "servo_stale", "servo_fatal"])
def test_servo_missing_or_unsafe_state_stops(fault):
    h=harness()
    if fault=="tf":h._push_fresh_transform=Mock(side_effect=ValueError("missing marker"))
    if fault=="joints":h._live_input_error=lambda:"joint state stale"
    if fault=="servo_stale":h.latest_servo_status_ns=1
    if fault=="servo_fatal":h.latest_servo_status=2
    SimplePolicyControlNode._push_servo_tick(h)
    h._complete_episode.assert_not_called()
    h._fail.assert_called_once()
    assert h._log_phase_event.call_args.kwargs["completion_reason"]=="SAFETY_STOP"
    np.testing.assert_array_equal(h._publish_twist.call_args.args[0], np.zeros(6))


def test_stale_seated_marker_aborts_after_zero_and_terminal_rollout():
    h = harness(book=seated_book())
    order = []
    h._publish_twist = lambda v: order.append(("twist", v.tolist()))
    def fail(reason):
        order.append(("failed", reason))
        h.phase = "holding_visualization"
    h._fail = fail
    h._publish_status = lambda phase, reason: order.append((phase, h.phase))
    h._push_fresh_transform = Mock(side_effect=ValueError("book pose stale: age=0.274s"))
    SimplePolicyControlNode._push_servo_tick(h)
    assert order[0] == ("twist", [0.] * 6)
    assert order[1][0] == "failed"
    assert order[2] == ("push_aborted", "holding_visualization")
    h._complete_episode.assert_not_called()
    h.visualization_hold_deadline_ns = None
    # Both timers must stay inert, even if the marker returns later.
    SimplePolicyControlNode._timer_callback(h)
    SimplePolicyControlNode._continuous_policy_tick(h)
    assert len(order) == 3


def test_stale_success_geometry_is_never_accepted():
    monitor = PushCompletion(.1, 90., .25)
    with pytest.raises(ValueError, match="stale"):
        monitor.observe(metrics(seated_book()), 1_000_000_000, 1_274_000_000)
    assert monitor.success_samples == 0


def _enter_marker_wait(h, clock, tcp):
    h._now_ns = lambda: clock[0]
    h.latest_servo_status_ns = clock[0]
    h.push_previous_tcp_xyz = tcp[:3, 3].copy()

    def stale_book(frame, _age):
        if frame == "target_book_center":
            raise ValueError("PUSH target_book_center TF stale: age=0.282s")
        return tcp.copy(), clock[0] - 10_000_000

    h._push_fresh_transform = Mock(side_effect=stale_book)
    SimplePolicyControlNode._push_servo_tick(h)
    assert h.phase == "push_waiting_for_fresh_state"


def test_marker_dropout_holds_zero_and_three_unique_frames_resume_same_push():
    h = harness(book=seated_book())
    clock = [20_000_000_000]
    tcp = translated(np.eye(4), .052)
    _enter_marker_wait(h, clock, tcp)
    np.testing.assert_array_equal(h._publish_twist.call_args.args[0], np.zeros(6))
    assert h._fail.call_count == 0
    initial_path = h.push_tcp_path_m
    stamps = [21_000_000_000, 21_000_000_000, 21_040_000_000, 21_080_000_000]
    positions = [.0525, .0525, .0535, .0540]

    for index, (stamp, position) in enumerate(zip(stamps, positions)):
        clock[0] += 40_000_000
        h.latest_servo_status_ns = clock[0]
        tcp[0, 3] = position
        h._push_fresh_transform = Mock(side_effect=lambda frame, age, stamp=stamp:(
            seated_book() if frame == "target_book_center" else tcp.copy(),
            stamp if frame == "target_book_center" else clock[0] - 10_000_000,
        ))
        SimplePolicyControlNode._push_waiting_for_fresh_state_tick(h)
        np.testing.assert_array_equal(h._publish_twist.call_args.args[0], np.zeros(6))
        if index == 1:
            assert h.push_recovery_fresh_samples == 1  # Cached duplicate ignored.
            assert h.phase == "push_waiting_for_fresh_state"

    assert h.phase == "push"
    assert h.target_eef is None  # Wait for a newly computed policy target.
    assert h.push_recovery_fresh_samples == 3
    assert h.push_completion.success_samples == 0  # Seated-looking recovery is not success.
    assert h._complete_episode.call_count == 0
    assert h.push_tcp_path_m == pytest.approx(initial_path + .002)
    recovered = [c for c in h._log_phase_event.call_args_list
                 if c.args[0] == "push_fresh_state_recovered"]
    assert len(recovered) == 1


def test_prolonged_marker_dropout_uses_existing_push_abort_path():
    h = harness()
    clock = [20_000_000_000]
    tcp = translated(np.eye(4), .052)
    _enter_marker_wait(h, clock, tcp)
    clock[0] = h.push_wait_start_ns + 1_000_000_000
    h.latest_servo_status_ns = clock[0]
    SimplePolicyControlNode._push_waiting_for_fresh_state_tick(h)
    h._fail.assert_called_once()
    h._complete_episode.assert_not_called()
    np.testing.assert_array_equal(h._publish_twist.call_args.args[0], np.zeros(6))
    assert h._publish_status.call_args.args[0] == "push_aborted"
    stopped = [c for c in h._log_phase_event.call_args_list if c.args[0] == "push_stopped"]
    assert stopped[-1].kwargs["completion_reason"] == "SAFETY_STOP"
