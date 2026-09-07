import ast
import importlib.util
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PROFILE_PATH = REPO / "scripts" / "sb3" / "targeted_dr_profiles.py"
ENV_PATH = REPO / "source" / "bookshelf" / "bookshelf" / "tasks" / "direct" / "bookshelf" / "bookshelf_residual_env.py"
CFG_PATH = REPO / "source" / "bookshelf" / "bookshelf" / "tasks" / "direct" / "bookshelf" / "bookshelf_residual_env_cfg.py"
BASE_ENV_PATH = REPO / "source" / "bookshelf" / "bookshelf" / "tasks" / "direct" / "bookshelf" / "bookshelf_env_v4.py"
SWEEP_PATH = REPO / "scripts" / "hpc" / "bookshelf_fresh_release_sweep.tsv"


def _profiles_module():
    spec = importlib.util.spec_from_file_location("targeted_dr_profiles", PROFILE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCfg:
    enable_policy_book_observation_bias = False
    enable_insert_gripper_observation_nuisance = False
    insert_gripper_observation_deterministic_values = ()
    enable_insert_action_realization_dr = False
    enable_targeted_dr_curriculum = False
    enable_residual_reset_curriculum = True
    book_grasp_translation_jitter_min = None
    book_grasp_translation_jitter_max = None
    book_grasp_x_jitter = 0.003
    book_grasp_y_jitter = 0.003
    book_grasp_z_jitter = 0.0015
    book_grasp_yaw_jitter = math.radians(3.0)
    insert_action_realization_scale_min = 1.0
    insert_action_realization_scale_max = 1.0


def test_disabled_profile_is_an_exact_noop():
    profiles = _profiles_module()
    cfg = FakeCfg()
    before = dict(vars(cfg))
    assert profiles.apply_targeted_dr_profile(cfg, "disabled") == {"profile": "disabled"}
    assert vars(cfg) == before


def test_requested_profiles_are_independent_and_bounded():
    profiles = _profiles_module()
    obs = FakeCfg()
    profiles.apply_targeted_dr_profile(obs, "obs_moderate")
    assert obs.policy_book_observation_translation_bias_max == (0.005, 0.010, 0.004)
    assert obs.book_grasp_translation_jitter_max == (0.003, 0.003, 0.0015)
    assert (obs.insert_gripper_observation_min, obs.insert_gripper_observation_max) == (0.0, 0.45)
    assert not obs.enable_insert_action_realization_dr

    combined = FakeCfg()
    profiles.apply_targeted_dr_profile(combined, "combined_moderate")
    assert combined.book_grasp_translation_jitter_max == (0.005, 0.008, 0.003)

    strong = FakeCfg()
    profiles.apply_targeted_dr_profile(strong, "strong_tail")
    assert strong.policy_book_observation_translation_bias_max == (0.007, 0.012, 0.006)
    assert strong.book_grasp_translation_jitter_max == (0.010, 0.010, 0.005)

    dynamics = FakeCfg()
    profiles.apply_targeted_dr_profile(dynamics, "combined_moderate_actuation")
    assert dynamics.enable_insert_action_realization_dr
    assert (dynamics.insert_action_realization_scale_min, dynamics.insert_action_realization_scale_max) == (0.65, 0.90)

    curriculum = FakeCfg()
    profiles.apply_targeted_dr_profile(curriculum, "obs_moderate_curriculum")
    assert curriculum.enable_targeted_dr_curriculum
    assert curriculum.policy_book_observation_translation_bias_max == (0.005, 0.010, 0.004)

    simple = FakeCfg()
    profiles.apply_targeted_dr_profile(simple, "obs_gripper_no_grasp")
    assert simple.book_grasp_translation_jitter_min == (0.0, 0.0, 0.0)
    assert simple.book_grasp_translation_jitter_max == (0.0, 0.0, 0.0)
    assert simple.book_grasp_yaw_jitter == 0.0


def test_physical_like_evaluation_bias_is_fixed_and_matches_measured_sign():
    profiles = _profiles_module()
    cfg = FakeCfg()
    profiles.apply_evaluation_dr_profile(cfg, "physical_like")
    expected_xyz = (0.0, -0.007, 0.0)
    expected_rpy = (0.0, 0.0, math.radians(-6.0))
    assert cfg.policy_book_observation_translation_bias_min == expected_xyz
    assert cfg.policy_book_observation_translation_bias_max == expected_xyz
    assert cfg.policy_book_observation_rpy_bias_min == expected_rpy
    assert cfg.policy_book_observation_rpy_bias_max == expected_rpy
    assert cfg.insert_gripper_observation_deterministic_values == (0.388235,)


def test_gripper_only_evaluation_includes_exact_physical_value():
    profiles = _profiles_module()
    cfg = FakeCfg()
    profiles.apply_evaluation_dr_profile(cfg, "gripper_only")
    assert 0.388235 in cfg.insert_gripper_observation_deterministic_values
    assert min(cfg.insert_gripper_observation_deterministic_values) == 0.0
    assert max(cfg.insert_gripper_observation_deterministic_values) == 0.45


def test_observation_bias_is_not_used_by_reward_or_ground_truth_metrics():
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    methods = {
        node.name: ast.get_source_segment(ENV_PATH.read_text(encoding="utf-8"), node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    bias_token = "_policy_book_translation_bias_env"
    assert bias_token in methods["_get_observations"]
    assert bias_token not in methods["_get_rewards"]
    base_tree = ast.parse(BASE_ENV_PATH.read_text(encoding="utf-8"))
    ground_truth_metrics = next(
        ast.get_source_segment(BASE_ENV_PATH.read_text(encoding="utf-8"), node)
        for node in ast.walk(base_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_task_metrics"
    )
    assert bias_token not in ground_truth_metrics


def test_release_objective_is_training_reward_only_and_does_not_gate_release():
    source = ENV_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    token = "release_training_objective"
    assert token in methods["_get_rewards"]
    assert token not in methods["_pre_physics_step"]
    assert token not in methods["_get_dones"]
    assert "release_never_timeout_penalty" in methods["_get_rewards"]


def test_six_action_twelve_observation_contract_is_unchanged():
    cfg = CFG_PATH.read_text(encoding="utf-8")
    env = ENV_PATH.read_text(encoding="utf-8")
    assert "enable_base_y_rotation_action = False" in cfg
    assert "expected_action_space = 7 if base_y_rotation_enabled else 6" in env
    assert "policy_obs[:, 11]" in env


def test_fresh_sweep_has_five_distinct_one_seed_formulations():
    rows = [line.split("\t") for line in SWEEP_PATH.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        ["F0", "obs_moderate", "original", "42"],
        ["F1", "obs_moderate", "smooth_readiness", "42"],
        ["F2", "obs_moderate_curriculum", "smooth_readiness", "42"],
        ["F3", "obs_gripper_no_grasp", "smooth_readiness", "42"],
        ["F4", "combined_moderate", "smooth_readiness", "42"],
    ]


def test_targeted_dr_curriculum_scales_from_mild_to_full():
    initial = 0.25
    total = 30_000_000
    scale = lambda step: initial + (1.0 - initial) * min(1.0, max(0.0, step / total))
    assert scale(0) == 0.25
    assert scale(15_000_000) == 0.625
    assert scale(30_000_000) == 1.0
