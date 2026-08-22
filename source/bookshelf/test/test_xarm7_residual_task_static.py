from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "source/bookshelf/bookshelf/tasks/direct/bookshelf"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_xarm7_task_is_registered_without_replacing_panda_task():
    registration = read(TASK / "__init__.py")
    assert 'id="Bookshelf-Residual-Direct-v0"' in registration
    assert 'id="Bookshelf-XArm7-Residual-Direct-v0"' in registration
    assert "bookshelf_xarm7_residual_env_cfg:BookshelfEnvCfg" in registration


def test_xarm7_config_uses_physical_names_and_tcp_offset():
    config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    for expected in (
        'robot_arm_joint_names_expr = "joint[1-7]"',
        "robot_finger_joint_names_expr = XARM_GRIPPER_STATE_JOINT_EXPR",
        "robot_gripper_command_joint_names_expr = XARM_GRIPPER_COMMAND_JOINT_EXPR",
        'robot_left_finger_body_name = "left_finger"',
        'robot_right_finger_body_name = "right_finger"',
        'robot_ee_body_name = "link7"',
        'robot_grasp_frame_body_name = ""',
        "ik_body_offset_pos = (0.0, 0.0, 0.172)",
        "debug_freeze_tool_to_book_transform = False",
    ):
        assert expected in config


def test_xarm7_uses_official_usd_and_commands_only_the_drive_joint():
    config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    asset = read(TASK / "xarm7_asset_cfg.py")
    assert "sim_utils.UsdFileCfg(" in asset
    assert 'Robots/Ufactory/xarm7/xarm7.usd' in asset
    assert 'os.environ.get("BOOKSHELF_XARM7_USD_PATH"' in asset
    assert "func=_spawn_xarm7_from_usd" in asset
    assert 'nested_root_path = f"{prim_path}/gripper/root_joint"' in asset
    assert "RemoveAPI(UsdPhysics.ArticulationRootAPI)" in asset
    assert 'articulation_root_prim_path="/root_joint"' in asset
    assert 'XARM_GRIPPER_COMMAND_JOINT_EXPR = "drive_joint"' in asset
    assert "XARM7_FINGER_INNER_SURFACE_OFFSET_M = 0.0260032" in asset
    assert "XARM7_SIM_BOOK_THICKNESS_M = 0.034" in asset
    assert "def xarm7_gripper_joint_for_pad_gap" in asset
    assert "XARM7_SIM_BOOK_SPAWN_JOINT_POS = xarm7_gripper_joint_for_pad_gap(" in asset
    assert "XARM7_SIM_BOOK_HOLD_JOINT_POS = xarm7_gripper_joint_for_pad_gap(0.032)" in asset
    for joint_name in (
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    ):
        assert f'"{joint_name}": XARM7_SIM_BOOK_SPAWN_JOINT_POS' in config
        assert joint_name in asset
    assert "joint_names_expr=[XARM_GRIPPER_COMMAND_JOINT_EXPR]" in asset
    assert '"joint[6-7]": 1200.0' in asset
    assert '"joint[6-7]": 120.0' in asset
    assert "gripper_closed_joint_pos = XARM7_SIM_BOOK_HOLD_JOINT_POS" in config
    assert "gripper_push_closed_joint_pos = XARM7_GRIPPER_FULLY_CLOSED_JOINT_POS" in config
    assert "script_retreat_steps = 60" in config
    assert "script_close_steps = 30" in config
    assert "script_retreat_dx = -0.001" in config


def test_xarm7_task_preserves_original_bookshelf_and_book_conventions():
    config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    for expected in (
        "book_grasp_offset_hand = (0.0, 0.0, 0.075)",
        "book_standing_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)",
        'book_grasp_orientation_source = "grasp_relative"',
        "slot_center_y = 0.0",
        "slot_x_open = 0.63",
        "slot_x_back = 0.83",
        "shelf_top_z = XARM7_SHELF_TOP_Z",
    ):
        assert expected in config


def test_drop_threshold_is_phase_aware_and_shared_with_reset_acceptance():
    base_config = read(TASK / "bookshelf_env_cfg_v4.py")
    panda_config = read(TASK / "bookshelf_residual_env_cfg.py")
    xarm_config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    shared = read(TASK / "bookshelf_env_v4.py")
    v5 = read(TASK / "bookshelf_env_v5.py")

    assert "BOOK_TRUE_GROUND_LOWEST_Z_THRESH = 0.002" in base_config
    assert "book_true_ground_lowest_z_thresh = BOOK_TRUE_GROUND_LOWEST_Z_THRESH" in base_config
    assert "reset_acceptance_ground_height_m = BOOK_TRUE_GROUND_LOWEST_Z_THRESH" in panda_config
    assert "reset_acceptance_ground_height_m = BOOK_TRUE_GROUND_LOWEST_Z_THRESH" in xarm_config
    assert "def _book_dropped_for_mode" in shared
    assert "self._book_dropped_for_mode(mode_start)" in shared
    assert "self._book_dropped_for_mode(mode_before)" in shared
    assert "self._book_dropped_for_mode(mode_before)" in v5
    assert "(failure_code == _DONE_NONE) & ~(mode_before == _MODE_PUSH)" in shared
    assert "(failure_code == _DONE_NONE) & ~(mode_before == _MODE_PUSH)" in v5


def test_xarm7_reset_aligns_book_thickness_to_the_gripper():
    shared = read(TASK / "bookshelf_env_v4.py")
    base_config = read(TASK / "bookshelf_env_cfg_v4.py")
    assert 'book_grasp_orientation_source = "world_standing"' in base_config
    assert 'if source == "grasp_relative":' in shared
    assert "self._book_grasp_relative_quat" in shared
    assert '"book_half_extent_across_gripper_m"' in shared


def test_xarm7_base_is_grounded_and_has_visual_references():
    config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    residual = read(TASK / "bookshelf_residual_env.py")
    assert "XARM7_SCENE_BASE_POS" in config
    assert "XARM7_APPROVED_SLOT_CENTER_BASE" in config
    assert "XARM7_REVIEWED_PRETARGET_TCP_BASE" in config
    assert "XARM7_SIM_SLOT_CENTER[0] - XARM7_APPROVED_SLOT_CENTER_BASE[0]" in config
    assert "XARM7_SIM_SLOT_CENTER[1]," in config
    assert "centering the simulated ten-slot row laterally on the robot base" in config
    assert "    0.0," in config
    assert "pos=XARM7_SCENE_BASE_POS" in config
    assert "robot_base_reference_pos = XARM7_SCENE_BASE_POS" in config
    assert "show_robot_base_reference_marker = True" in config
    assert "show_target_book_marker = True" in config
    assert "def _create_robot_base_reference_marker" in residual
    assert 'radius=0.09' in residual


def test_xarm7_physical_pretarget_uses_the_official_usd_joint_branch():
    config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    asset = read(TASK / "xarm7_asset_cfg.py")
    for source in (config, asset):
        assert '"joint3": 4.904658882462919 - 2.0 * math.pi' in source
        assert '"joint5": 3.302595179623167 - 2.0 * math.pi' in source
        assert '"joint7": 4.4791192150828865 - 2.0 * math.pi' in source


def test_xarm7_reset_solves_the_reviewed_slot_relative_tcp_pose_with_ik():
    config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    base_config = read(TASK / "bookshelf_residual_env_cfg.py")
    residual = read(TASK / "bookshelf_residual_env.py")
    assert "reset_to_slot_relative_tool_pose = False" in base_config
    assert "reset_to_slot_relative_tool_pose = True" in config
    assert "reset_tool_offset_slot_xyz" in config
    assert "reset_tool_quaternion_slot_wxyz" in config
    assert "reset_tool_offset_slot_xyz = XARM7_PRETARGET_OFFSET_SLOT" in config
    assert "reset_tool_quaternion_slot_wxyz = XARM7_REVIEWED_PRETARGET_TCP_QUAT_WXYZ" in config
    assert "shelf_top_z = XARM7_SHELF_TOP_Z" in config
    assert "def _reset_to_slot_relative_tool_pose" in residual
    assert "self._compute_ik_joint_targets_from_tool_quat(" in residual
    assert '"[XARM_RANDOMIZED_RESET] "' in residual


def test_ik_converts_scene_targets_to_the_shifted_robot_base_frame():
    shared = read(TASK / "bookshelf_env_v4.py")
    residual = read(TASK / "bookshelf_residual_env.py")
    v5 = read(TASK / "bookshelf_env_v5.py")
    assert "def _position_env_to_base" in shared
    assert "self.robot.data.root_pos_w - self.scene.env_origins" in shared
    assert "math_utils.quat_apply_inverse(" in shared
    assert "target_pos_b = self._position_env_to_base(target_pos_env)" in shared
    assert "target_pos_b = self._position_env_to_base(target_pos_env)" in residual
    assert v5.count("target_pos_b = self._position_env_to_base(target_pos_env)") == 2


def test_zero_agent_can_hold_the_reset_pose_for_visual_inspection():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    assert '"--freeze_nominal_controller"' in zero_agent
    assert "env_cfg.debug_freeze_nominal_controller = True" in zero_agent


def test_zero_agent_has_a_direct_reachable_xarm_grasp_demo():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    assert '"--xarm_reachable_grasp_demo"' in zero_agent
    assert '"--xarm_panda_reset_grasp_demo"' in zero_agent
    assert '"--xarm_grasp_hold_gap_mm"' in zero_agent
    assert "env_cfg.gripper_push_closed_joint_pos = hold_joint_pos" not in zero_agent
    assert 'f"drive_joint={hold_joint_pos:.6f} rad; PUSH still closes fully"' in zero_agent
    assert "xarm7_gripper_joint_for_pad_gap" in zero_agent
    assert "env_cfg.gripper_closed_joint_pos = hold_joint_pos" in zero_agent
    assert "env_cfg.reset_to_slot_relative_tool_pose = False" in zero_agent
    assert "env_cfg.debug_hold_book_fixed_to_tool = False" in zero_agent
    assert "env_cfg.debug_robot_target_pose_only = True" in zero_agent
    assert "env_cfg.debug_spawn_book_with_collision_clearance = bool(" in zero_agent
    assert "env_cfg.debug_reachable_grasp_sequence = False" in zero_agent
    assert "env_cfg.debug_freeze_nominal_controller = True" in zero_agent
    assert "env_cfg.debug_omit_bookshelf_obstacles = True" in zero_agent
    assert "env_cfg.show_reachable_grasp_target_frame = True" in zero_agent
    assert 'env_cfg.reachable_grasp_target_frame_source = "sequence_target"' in zero_agent
    assert '"[XARM_GRASP_DEMO] The verified xArm target pose and book-matched "' in zero_agent
    assert '"measured finger pads with palm clearance."' in zero_agent
    assert '"[XARM_GRASP_DEMO] smooth gripper closure ramp: "' in zero_agent


def test_xarm_can_compare_against_the_panda_reset_grasp():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    residual_config = read(TASK / "bookshelf_residual_env_cfg.py")
    residual = read(TASK / "bookshelf_residual_env.py")

    assert '"--xarm_panda_reset_grasp_demo"' in zero_agent
    assert "env_cfg.debug_spawn_book_panda_style = bool(" in zero_agent
    assert "env_cfg.debug_robot_target_gripper_ramp_steps = 1" in zero_agent
    assert "env_cfg.debug_robot_target_gripper_settle_steps = 10" in zero_agent
    assert '"[XARM_PANDA_RESET_GRASP_DEMO] The xArm is spawned at the "' in zero_agent
    assert "debug_spawn_book_panda_style = False" in residual_config
    assert 'getattr(self.cfg, "debug_spawn_book_panda_style", False)' in residual
    assert "def _spawn_book_panda_style" in residual
    assert "book_state = self._snap_book_to_measured_grasp(env_ids_t)" in residual
    assert '"[XARM_PANDA_RESET] book snapped to measured finger midpoint; "' in residual

    assert "show_reachable_grasp_target_frame = False" in residual_config
    assert 'reachable_grasp_target_frame_source = "slot_relative"' in residual_config
    assert "debug_snap_book_to_grasp_on_reset = False" in residual_config
    assert "debug_reachable_grasp_sequence = False" in residual_config
    assert "def _create_reachable_grasp_target_frame" in residual
    assert '"/World/Visuals/XArmReachableGraspTargetFrame"' in residual
    assert "self.cfg.reset_tool_offset_slot_xyz" in residual
    assert "self.cfg.reset_tool_quaternion_slot_wxyz" in residual
    assert 'if source == "current_tool"' in residual
    assert 'source == "sequence_target"' in residual
    assert "self._ik_body_offset_pos_b[0:1]" in residual
    assert 'getattr(self.cfg, "debug_snap_book_to_grasp_on_reset", False)' in residual
    assert "self._snap_book_to_measured_grasp(env_ids_t)" in residual
    assert "def _prepare_reachable_grasp_sequence" in residual
    assert "def _apply_reachable_grasp_sequence" in residual
    assert "def _align_gripper_to_sampled_slot" in residual
    assert 'getattr(self.cfg, "debug_reachable_grasp_sequence", False)' in residual
    assert "super()._align_gripper_to_sampled_slot(env_ids_t)" in residual
    assert "parked_book_state[:, 2] -= 5.0" in residual
    assert "self._snap_book_to_measured_grasp(env_ids_t)" in residual
    assert "no later step rewrites its root state" in residual
    assert "if self._apply_reachable_grasp_sequence():" in residual
    assert '"[XARM_GRASP_SEQUENCE] dynamic book spawned in settled near-contact "' in residual
    assert '"gripper; holding the measured grasp width"' in residual
    assert '"[XARM_GRASP_SEQUENCE] constant-width dynamic grasp observation complete; "' in residual
    assert "closed_value - open_value" not in residual
    assert "fixed_book_mask = close_mask" not in residual
    assert "def _spawn_book_with_collision_clearance" in residual
    assert "book_half_extent_grasp[:, 2] + palm_clearance" in residual
    assert "book would overlap the xArm finger collision surfaces" in residual
    assert '"[XARM_BOOK_CLEARANCE] book placed without changing "' in residual


def test_xarm_can_move_a_dynamic_grasp_forward_and_back():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    residual_config = read(TASK / "bookshelf_residual_env_cfg.py")
    residual = read(TASK / "bookshelf_residual_env.py")

    assert '"--xarm_forward_backward_demo"' in zero_agent
    assert '"--xarm_motion_distance_mm"' in zero_agent
    assert '"--xarm_motion_half_period_steps"' in zero_agent
    assert "env_cfg.debug_robot_forward_backward_demo = bool(" in zero_agent
    assert '"[XARM_FORWARD_BACKWARD_DEMO] after the dynamic grasp "' in zero_agent
    assert "env_cfg.debug_omit_bookshelf_obstacles = False" in zero_agent
    assert '4 if args_cli.missing_index is None else int(args_cli.missing_index)' in zero_agent
    assert "env_cfg.forced_missing_book_index = selected_missing_index" in zero_agent
    assert "0.5 * row_pitch_m" in zero_agent
    assert "if args_cli.missing_index is None and not args_cli.all_missing_indices" in zero_agent
    assert "else 0.0" in zero_agent
    assert "selected_slot_center_y_m = (" in zero_agent
    assert "debug_row_layout_y_offset_m = 0.0" in residual_config
    assert "def _apply_debug_row_layout_y_offset" in residual
    assert "self._slot_center_y_env[env_ids_t] += offset_m" in residual
    assert '"[XARM_BOOKSHELF_DEBUG] shifted the complete side-book row "' in residual
    assert "debug_robot_forward_backward_demo = False" in residual_config
    assert "debug_robot_forward_backward_distance_m = 0.05" in residual_config
    assert "debug_robot_forward_backward_wait_steps = 60" in residual_config
    assert "debug_robot_forward_backward_half_period_steps = 180" in residual_config
    assert "def _apply_robot_forward_backward_motion" in residual
    assert "support_released = self._robot_target_gripper_ramp_step > int(hold_steps)" in residual
    assert "0.5 * distance_m * (1.0 - torch.cos(phase))" in residual
    assert "moving_target_pos[:, 0] += displacement[moving]" in residual
    assert '"[XARM_FORWARD_BACKWARD] dynamic grasp settled; starting "' in residual


def test_xarm_can_hand_a_dynamic_grasp_to_the_nominal_controller():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    residual_config = read(TASK / "bookshelf_residual_env_cfg.py")
    xarm_config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    residual = read(TASK / "bookshelf_residual_env.py")

    assert '"--xarm_nominal_controller_demo"' in zero_agent
    assert '"--all_missing_indices"' in zero_agent
    assert "--all_missing_indices and --missing_index cannot be used together" in zero_agent
    assert "env_cfg.debug_forced_missing_book_index_sequence = tuple(" in zero_agent
    assert "args_cli.episodes = row_book_count" in zero_agent
    assert '"--xarm_nominal_retreat_mm"' in zero_agent
    assert '"--xarm_nominal_push_step_mm"' in zero_agent
    assert '"--xarm_nominal_episode_length_s"' in zero_agent
    assert '"--xarm_nominal_push_vertical_target_lead_mm"' in zero_agent
    assert '"--xarm_nominal_push_target_lead_mm"' in zero_agent
    assert '"--xarm_nominal_push_vertical_step_mm"' in zero_agent
    assert '"--xarm_nominal_push_recovery_step_mm"' in zero_agent
    assert '"--xarm_shelf_closer_mm"' in zero_agent
    assert "if args_cli.xarm_shelf_closer_mm is None" in zero_agent
    assert "50.0" in zero_agent
    assert "args_cli.xarm_shelf_closer_mm < 0.0" in zero_agent
    assert "env_cfg.slot_x_open = float(env_cfg.slot_x_open) - shelf_shift_m" in zero_agent
    assert "env_cfg.slot_x_back = float(env_cfg.slot_x_back) - shelf_shift_m" in zero_agent
    assert "env_cfg.debug_robot_nominal_controller_demo = bool(" in zero_agent
    assert "env_cfg.enable_nominal_release_assist = True" in zero_agent
    assert "env_cfg.episode_length_s = float(" in zero_agent
    assert "args_cli.xarm_nominal_episode_length_s" in zero_agent
    assert "env_cfg.nominal_release_assist_until_frac = 1.0" in zero_agent
    assert "env_cfg.nominal_push_dx = (" in zero_agent
    assert "0.001 * float(args_cli.xarm_nominal_push_step_mm)" in zero_agent
    assert "env_cfg.nominal_push_dz_limit = (" in zero_agent
    assert "0.001 * float(args_cli.xarm_nominal_push_vertical_step_mm)" in zero_agent
    assert "env_cfg.debug_use_full_target_ee_quat = False" in zero_agent
    assert "env_cfg.debug_use_base_frame_quat_deltas = False" in zero_agent
    assert "env_cfg.debug_position_only_target_ee = True" in zero_agent
    assert "env_cfg.debug_pose_ik_rotation_weight = 1.0" in zero_agent
    assert "env_cfg.debug_integrate_position_target_ee = True" in zero_agent
    assert "env_cfg.debug_scripted_current_relative_target = False" in zero_agent
    assert "env_cfg.debug_scripted_fixed_retreat_path = True" in zero_agent
    assert "-0.001 * float(args_cli.xarm_nominal_retreat_mm)" in zero_agent
    assert "env_cfg.debug_nominal_push_current_relative_target = False" in zero_agent
    assert "env_cfg.debug_nominal_push_reuse_insert_forward = True" in zero_agent
    assert "env_cfg.debug_nominal_push_lower_before_forward = True" in zero_agent
    assert "env_cfg.debug_nominal_push_align_to_book_center = True" in zero_agent
    assert "env_cfg.debug_nominal_push_hold_y_only = True" in zero_agent
    assert "env_cfg.debug_nominal_push_lock_y_to_entry = True" in zero_agent
    assert "env_cfg.debug_nominal_push_max_target_lead_m = (" in zero_agent
    assert "0.001 * float(args_cli.xarm_nominal_push_target_lead_mm)" in zero_agent
    assert "env_cfg.debug_nominal_push_max_vertical_target_lead_m = (" in zero_agent
    assert "float(args_cli.xarm_nominal_push_vertical_target_lead_mm)" in zero_agent
    assert "env_cfg.debug_nominal_push_tracking_pause_enabled = False" in zero_agent
    assert "env_cfg.debug_nominal_push_spine_tracking_enabled = False" in zero_agent
    assert "env_cfg.debug_nominal_push_spine_recovery_step_m = (" in zero_agent
    assert "0.001 * float(args_cli.xarm_nominal_push_recovery_step_mm)" in zero_agent
    assert "env_cfg.debug_robot_nominal_handoff_wait_steps = 0" in zero_agent
    assert "args_cli.xarm_nominal_controller_demo" in zero_agent
    assert "env_cfg.debug_position_only_target_ee = not bool(" not in zero_agent
    assert "env_cfg.debug_spawn_inside_fraction = 0.0" in zero_agent
    assert '"[XARM_NOMINAL_CONTROLLER_DEMO] use the original "' in zero_agent
    assert '"measured pre-insertion orientation held fixed, retained "' in zero_agent
    assert '"Cartesian INSERT, fixed straight-line retreat, and a "' in zero_agent
    assert '"book-centered lowering before a bounded retained-target "' in zero_agent
    assert '"Panda PUSH along +X only while the gripper is fully closed, "' in zero_agent
    assert "debug_robot_nominal_controller_demo = False" in residual_config
    assert "debug_forced_missing_book_index_sequence = None" in residual_config
    assert "debug_use_base_frame_quat_deltas = False" in residual_config
    assert "debug_pose_ik_rotation_weight = None" in residual_config
    assert "debug_nominal_push_max_vertical_target_lead_m = None" in residual_config
    assert "debug_integrate_position_target_ee = False" in residual_config
    assert "debug_scripted_current_relative_target = False" in residual_config
    assert "debug_scripted_fixed_retreat_path = False" in residual_config
    assert "debug_scripted_fixed_retreat_total_dx = None" in residual_config
    assert "script_retreat_steps = 60" in xarm_config
    assert "script_retreat_dx = -0.001" in xarm_config
    assert "debug_scripted_fixed_retreat_total_dx = -0.120" in xarm_config
    assert "debug_nominal_push_current_relative_target = False" in residual_config
    assert "debug_nominal_push_reuse_insert_forward = False" in residual_config
    assert "debug_nominal_push_lower_before_forward = False" in residual_config
    assert "debug_nominal_push_align_to_book_center = False" in residual_config
    assert "debug_nominal_push_hold_y_only = False" in residual_config
    assert "debug_nominal_push_lock_y_to_entry = False" in residual_config
    assert "debug_nominal_push_max_target_lead_m = None" in residual_config
    assert "debug_nominal_push_tracking_pause_enabled = False" in residual_config
    assert "debug_nominal_push_tracking_pause_joint_error_rad = 0.12" in residual_config
    assert "debug_nominal_push_tracking_resume_joint_error_rad = 0.08" in residual_config
    assert "debug_nominal_push_spine_tracking_enabled = False" in residual_config
    assert "debug_nominal_push_spine_pause_lateral_error_m = 0.004" in residual_config
    assert "debug_nominal_push_spine_resume_lateral_error_m = 0.002" in residual_config
    assert "debug_nominal_push_spine_recovery_step_m = 0.002" in residual_config
    assert "debug_robot_nominal_handoff_wait_steps = 60" in residual_config
    assert "def _prepare_robot_nominal_preinsert_pose" in residual
    assert "self._debug_missing_index_sequence_cursor = 0" in residual
    assert '"[XARM_SLOT_SEQUENCE] "' in residual
    assert "self._spawn_at_planned_tool_pose(env_ids_t)" in residual
    assert "self._target_book_pose_tensors(inside_fraction=0.0)" in residual
    assert "joint_pos[:, self._gripper_command_joint_ids]" in residual
    assert "def _place_ideal_book_at_nominal_target" in residual
    assert 'getattr(self.cfg, "debug_hold_book_fixed_to_tool", False)' in residual
    assert "self._place_ideal_book_at_nominal_target(handoff_ids)" in residual
    assert '"[XARM_NOMINAL_PREINSERT] original book-relative target "' in residual
    assert "def _apply_robot_nominal_controller_handoff" in residual
    assert "def _apply_base_frame_orientation_delta" in residual
    assert 'getattr(self.cfg, "debug_pose_ik_rotation_weight", None)' in residual
    assert "tool_position_jacobian = (" in residual
    assert "offset_skew @ jacobian[:, 3:6, :]" in residual
    assert "torch.linalg.solve(" in residual
    assert 'getattr(self.cfg, "debug_use_base_frame_quat_deltas", False)' in residual
    assert 'getattr(self.cfg, "debug_integrate_position_target_ee", False)' in residual
    assert 'getattr(self.cfg, "debug_scripted_current_relative_target", False)' in residual
    assert "target_pos_env[scripted_mask] = ee_tool_pos_env[scripted_mask]" in residual
    assert 'getattr(self.cfg, "debug_scripted_fixed_retreat_path", False)' in residual
    assert '"debug_scripted_fixed_retreat_total_dx", None' in residual
    assert "self._debug_scripted_retreat_start_pos_env" in residual
    assert "retreat_progress * retreat_total_dx" in residual
    assert "target_pos_env[scripted_mask] = fixed_retreat_target[scripted_mask]" in residual
    assert 'getattr(self.cfg, "debug_nominal_push_current_relative_target", False)' in residual
    assert '"PANDA_BOUNDED_RETAINED"' in residual
    assert "Start PUSH from the measured post-retreat pose" in residual
    assert '"[XARM_PUSH_PHASE] CENTERING_AND_LOWERING; "' in residual
    assert '"[XARM_PUSH_PHASE] PUSHING_STRAIGHT_X; book-center Y and "' in residual
    assert "target_pos_env[push_mask] = ee_tool_pos_env[push_mask]" in residual
    assert 'getattr(self.cfg, "debug_nominal_push_reuse_insert_forward", False)' in residual
    assert 'getattr(self.cfg, "debug_nominal_push_lower_before_forward", False)' in residual
    assert "def _update_nominal_push_lowering_phase" in residual
    assert "self._debug_nominal_push_target_z_env" in residual
    assert "self._debug_nominal_push_lowering_complete" in residual
    assert "float(self.cfg.nominal_push_z_fraction_from_bottom) * book_height" in residual
    assert "self._debug_integrated_target_pos_env[:, 2]" in residual
    assert "nominal[:, 0] = torch.where(push_forward, insert_dx" in residual
    assert "nominal[:, 2] = torch.where(push_mask, insert_dz, nominal[:, 2])" not in residual
    assert 'getattr(self.cfg, "debug_nominal_push_hold_y_only", False)' in residual
    assert 'getattr(self.cfg, "debug_nominal_push_lock_y_to_entry", False)' in residual
    assert "self._debug_nominal_push_line_y_env" in residual
    assert "self._debug_nominal_push_line_initialized" in residual
    assert "target_pos_env[push_mask, 1] = self._debug_nominal_push_line_y_env[" in residual
    assert '"[XARM_PUSH_TARGET] "' in residual
    assert '"debug_nominal_push_max_target_lead_m", None' in residual
    assert "target_pos_env[push_mask, 0] = torch.minimum(" in residual
    assert "target_pos_env[push_mask, 2] = torch.clamp(" in residual
    assert "push_dy = torch.zeros_like(push_dy)" in residual
    assert "push_dz = torch.zeros_like(push_dz)" not in residual
    assert 'getattr(self.cfg, "debug_nominal_push_tracking_pause_enabled", False)' in residual
    assert "_wrap_to_pi(self._arm_hold_joint_pos - arm_joint_pos)" in residual
    assert "delta[paused, 0] = 0.0" in residual
    assert "delta[paused, 2] = 0.0" in residual
    assert "push_spine_y_error = ee_tool_pos_env[:, 1] - book_pos_env[:, 1]" in residual
    assert "self._debug_nominal_push_alignment_complete" in residual
    assert "alignment_lost = push_mask & alignment_complete" in residual
    assert "spine_recovery_dy = torch.clamp(" in residual
    assert "delta[push_mask, 1] = spine_recovery_dy[push_mask]" in residual
    assert "target_pos_env[push_mask, 1] = book_pos_env[push_mask, 1]" in residual
    assert "target_pos_env[paused, 0] = ee_tool_pos_env[paused, 0]" in residual
    assert 'f"push_x_paused={bool(' in residual
    assert 'f"push_aligned={bool(' in residual
    assert "self._debug_integrated_target_pos_env[:] = target_pos_env" in residual
    assert "delta[normal_mask, 3]" in residual
    assert "delta[normal_mask, 4]" in residual
    assert "self._debug_tool_to_book_transform_frozen[handoff_ids] = False" in residual
    assert "self._capture_fixed_tool_to_book_transform(handoff_ids)" in residual
    assert "self._mode[handoff_ids] = _MODE_INSERT" in residual
    assert "self._robot_nominal_handoff_complete[handoff_ids] = True" in residual
    assert '"[XARM_NOMINAL_HANDOFF] dynamic grasp measured; nominal "' in residual


def test_zero_agent_has_an_isolated_xarm_target_pose_demo():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    shared = read(TASK / "bookshelf_env_v4.py")
    residual_config = read(TASK / "bookshelf_residual_env_cfg.py")
    residual = read(TASK / "bookshelf_residual_env.py")

    assert '"--xarm_target_pose_demo"' in zero_agent
    assert "env_cfg.debug_robot_target_pose_only = True" in zero_agent
    assert "env_cfg.debug_reachable_grasp_sequence = False" in zero_agent
    assert "env_cfg.debug_omit_bookshelf_obstacles = True" in zero_agent
    assert "debug_robot_target_pose_only = False" in residual_config
    assert 'getattr(self.cfg, "debug_robot_target_pose_only", False)' in shared
    assert "This branch deliberately performs no physics step here" in shared
    assert "def _prepare_robot_target_pose_only" in residual
    assert "def _apply_robot_target_pose_only" in residual
    assert "parked_book_state[:, 2] -= 5.0" in residual
    assert "self.robot.data.default_joint_pos[env_ids_t].clone()" in residual
    assert "self._reachable_grasp_sequence_target_pos_w[env_ids_t] = target_tool_pos_w" in residual
    assert "self._reachable_grasp_sequence_target_quat_w[env_ids_t] = ee_body_quat_w" in residual
    assert '"[XARM_TARGET_POSE] robot state written directly; no physics "' in residual
    assert "if self._apply_robot_target_pose_only():" in residual


def test_xarm7_grasp_demo_starts_at_the_book_width():
    config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    assert "debug_reachable_grasp_preclose_joint_pos = XARM7_SIM_BOOK_SPAWN_JOINT_POS" in config
    assert "debug_finger_inner_surface_offset_m = XARM7_FINGER_INNER_SURFACE_OFFSET_M" in config
    assert "debug_book_min_finger_clearance_m = 0.0" in config
    assert "debug_robot_target_gripper_ramp_steps = 30" in config
    assert "debug_robot_target_gripper_settle_steps = 30" in config
    residual = read(TASK / "bookshelf_residual_env.py")
    assert 'float(self.cfg.gripper_closed_joint_pos)' in residual
    assert 'getattr(self.cfg, "debug_spawn_book_with_collision_clearance", False)' in residual
    assert "smooth_progress = progress * progress * (3.0 - 2.0 * progress)" in residual
    assert "self._robot_target_gripper_held_book_state" in residual
    assert '"[XARM_GRASP_DEMO] placement support released; the book is "' in residual
    assert "self._robot_target_gripper_ramp_step[env_ids_t] = 0" in residual
    assert "book_to_hand_quat_franka_axes_wxyz = (0.5, -0.5, -0.5, -0.5)" in read(
        TASK / "bookshelf_env_cfg_v4.py"
    )


def test_zero_agent_can_preserve_and_report_a_failed_grasp():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    residual_config = read(TASK / "bookshelf_residual_env_cfg.py")
    residual_env = read(TASK / "bookshelf_residual_env.py")
    shared_env = read(TASK / "bookshelf_env_v4.py")
    assert '"--debug_no_resets"' in zero_agent
    assert '"--debug_grasp_interval"' in zero_agent
    assert "env_cfg.debug_disable_episode_resets = True" in zero_agent
    assert "debug_grasp_snapshot(env_index=0)" in zero_agent
    assert "debug_disable_episode_resets = False" in residual_config
    assert 'getattr(self.cfg, "debug_disable_episode_resets", False)' in residual_env
    assert "terminated = torch.zeros_like(terminated)" in residual_env
    assert "time_out = torch.zeros_like(time_out)" in residual_env
    assert "def debug_grasp_snapshot" in shared_env
    assert '"robot_is_fixed_base": bool(self.robot.is_fixed_base)' in shared_env
    assert '"arm_joints": arm_joints' in shared_env
    assert '"arm_max_target_error_rad"' in shared_env
    assert '"finger_joints": finger_joints' in shared_env
    assert '"tool_position_env_m"' in shared_env


def test_zero_agent_can_stop_after_a_bounded_number_of_steps():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    assert '"--max_steps"' in zero_agent
    assert "if args_cli.max_steps < 0" in zero_agent
    assert "step_count >= args_cli.max_steps" in zero_agent
    assert 'f"[ZERO_AGENT] reached max_steps={args_cli.max_steps}; closing"' in zero_agent


def test_zero_agent_can_stop_and_report_each_completed_episode():
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    assert '"--episodes"' in zero_agent
    assert "if args_cli.episodes < 0" in zero_agent
    assert "_, _, terminated, truncated, info = env.step(actions)" in zero_agent
    assert '"[ZERO_AGENT_EPISODE] "' in zero_agent
    assert '"failure_name"' in zero_agent
    assert "completed_episodes >= args_cli.episodes" in zero_agent

    for filename in ("bookshelf_env_v4.py", "bookshelf_env_v5.py"):
        env = read(TASK / filename)
        assert 'self.extras["episode_metric_final_rear_to_mouth"] = rear_to_mouth.clone()' in env
        assert 'self.extras["episode_metric_final_front_to_back"] = front_to_back.clone()' in env
        assert 'self.extras["episode_metric_release_step"] = self._release_step_buf.clone()' in env


def test_xarm7_grasp_bounds_are_explicit_and_asymmetric_capable():
    shared = read(TASK / "bookshelf_env_v4.py")
    residual = read(TASK / "bookshelf_residual_env.py")
    config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    assert 'getattr(self.cfg, "book_grasp_translation_jitter_min", None)' in shared
    assert 'getattr(self.cfg, "book_grasp_translation_jitter_max", None)' in shared
    assert "residual_curriculum_grasp_translation_bounds_1" in config
    assert "residual_curriculum_grasp_translation_bounds_final" in config
    assert "self._capture_fixed_tool_to_book_transform(env_ids_t)" in residual


def test_xarm7_execution_keeps_seven_joint_ik_and_separate_gripper_commands():
    asset = read(TASK / "xarm7_asset_cfg.py")
    shared = read(TASK / "bookshelf_env_v4.py")
    residual = read(TASK / "bookshelf_residual_env.py")
    assert 'joint_names_expr=["joint[1-7]"]' in asset
    assert "self._compute_ik_joint_targets_from_tool" in shared
    assert "joint_ids=self._arm_joint_ids" in shared
    assert "joint_ids=self._arm_joint_ids" in residual
    assert "self._gripper_command_joint_ids" in shared
    assert "self._gripper_command_joint_ids" in residual


def test_success_depth_remains_valid_after_passing_the_minimum():
    config = read(TASK / "bookshelf_env_cfg_v4.py")
    manual_step = read(ROOT / "scripts/manual_step.py")

    assert "success_rear_to_mouth_min = -0.012" in config
    assert "success_rear_to_mouth_max" not in config
    assert "success_front_clear_min" not in config
    for filename in ("bookshelf_env_v4.py", "bookshelf_env_v5.py"):
        env = read(TASK / filename)
        assert "rear_ok = rear_to_mouth >= float(self.cfg.success_rear_to_mouth_min)" in env
        assert "rear_to_mouth <= float(self.cfg.success_rear_to_mouth_max)" not in env
        assert "front_ok = front_to_back <= float(self.cfg.success_front_clear_max) + front_eps" in env
        assert "front_to_back >= float(self.cfg.success_front_clear_min)" not in env

    assert "rear_ok = rear >= float(cfg.success_rear_to_mouth_min)" in manual_step
    assert "front_ok = front <= (float(cfg.success_front_clear_max) + front_eps)" in manual_step


def test_xarm_training_filters_invalid_randomized_grasps_only_before_ppo():
    shared_config = read(TASK / "bookshelf_residual_env_cfg.py")
    xarm_config = read(TASK / "bookshelf_xarm7_residual_env_cfg.py")
    residual = read(TASK / "bookshelf_residual_env.py")
    train = read(ROOT / "scripts/sb3/train.py")
    play = read(ROOT / "scripts/sb3/play.py")
    zero_agent = read(ROOT / "scripts/zero_agent.py")
    preflight = read(ROOT / "scripts/xarm_randomization_preflight.py")

    for config in (shared_config, xarm_config):
        assert "enable_reset_acceptance_gate = False" in config
        assert "reset_acceptance_validation_steps = 12" in config
        assert "reset_acceptance_max_attempts = 50" in config
        assert "reset_acceptance_translation_limit_m = 0.003" in config
        assert "reset_acceptance_rotation_limit_rad = math.radians(3.0)" in config
        assert "reset_acceptance_arm_error_limit_rad = math.radians(8.0)" in config

    assert '"--disable_reset_acceptance_gate"' in train
    assert '"--xarm_training_standoff_mm"' in train
    assert "default=30.0" in train
    assert 'args_cli.task == "Bookshelf-XArm7-Residual-Direct-v0"' in train
    assert "reset_offset[0] -= standoff_m" in train
    assert "env_cfg.xarm_training_reset_standoff_m = standoff_m" in train
    assert '"[XARM_TRAINING_RESET] additional_standoff_mm="' in train
    assert "xarm_training_reset_standoff_m = 0.030" in xarm_config
    assert "env_cfg.enable_reset_acceptance_gate = not bool(" in train
    for diagnostic_entrypoint in (play, zero_agent, preflight):
        assert "env_cfg.enable_reset_acceptance_gate = False" in diagnostic_entrypoint

    assert "def _validate_randomized_reset(" in residual
    assert "def _apply_reset_acceptance_gate(" in residual
    assert "self._reset_idx(rejected_ids)" in residual
    assert "randomized grasp reset failed the acceptance gate after" in residual
    assert 'log["reset_gate_acceptance_rate"]' in residual
    assert 'log[f"reset_gate_{reason}_total"]' in residual
    assert "self._capture_scenario_initial_pose(env_ids_t)" in residual


def test_training_can_compare_baseline_and_observable_geometry_release_guards():
    config = read(TASK / "bookshelf_residual_env_cfg.py")
    residual = read(TASK / "bookshelf_residual_env.py")
    train = read(ROOT / "scripts/sb3/train.py")

    assert 'policy_release_guard_mode = "none"' in config
    assert "premature_release_penalty = 0.5" in config
    assert '"--policy_release_guard"' in train
    assert 'choices=("none", "observable_geometry")' in train
    assert '"--premature_release_penalty"' in train
    assert "env_cfg.policy_release_guard_mode = str(args_cli.policy_release_guard)" in train
    assert "env_cfg.premature_release_penalty = premature_release_penalty" in train
    assert '"[POLICY_RELEASE_GUARD] mode="' in train

    assert 'self._policy_release_guard_mode == "observable_geometry"' in residual
    assert "geometry_ready = self._nominal_release_mask(metrics)" in residual
    assert "policy_release = raw_policy_release & geometry_ready" in residual
    assert "& (self._mode == _MODE_INSERT)" in residual
    assert "& ~geometry_ready" in residual
    assert 'self.extras["log"]["raw_policy_release_fraction"]' in residual
    assert 'self.extras["log"]["blocked_policy_release_fraction"]' in residual
    assert 'self.extras["log"]["premature_release_penalty_mean"]' in residual


def test_headless_training_disables_debug_visualization_markers():
    train = read(ROOT / "scripts/sb3/train.py")

    assert "if bool(args_cli.headless):" in train
    for marker_flag in (
        "show_robot_base_reference_marker",
        "show_target_book_marker",
        "show_target_ee_marker",
        "show_current_ee_marker",
        "show_reachable_grasp_target_frame",
    ):
        assert f'"{marker_flag}"' in train
    assert "setattr(env_cfg, marker_flag, False)" in train
    assert '"[HEADLESS_TRAINING] disabled debug visualization markers: "' in train
