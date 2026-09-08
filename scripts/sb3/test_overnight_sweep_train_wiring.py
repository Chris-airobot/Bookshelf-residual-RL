#!/usr/bin/env python3
"""AST and structural checks for train.py overnight sweep wiring."""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_PY = _REPO_ROOT / 'scripts' / 'sb3' / 'train.py'
_ENV_PY = (
    _REPO_ROOT
    / 'source'
    / 'bookshelf'
    / 'bookshelf'
    / 'tasks'
    / 'direct'
    / 'bookshelf'
    / 'bookshelf_residual_env.py'
)


def _get_train_tree() -> ast.AST:
    return ast.parse(_TRAIN_PY.read_text())


def test_overnight_variant_arg_and_task_gate():
    """Verify --overnight_variant exists and task is gated."""
    tree = _get_train_tree()

    found_arg = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument'
        ):
            if any(
                isinstance(arg, ast.Constant)
                and arg.value == '--overnight_variant'
                for arg in node.args
            ):
                found_arg = True
                choices_kw = next(
                    (kw for kw in node.keywords if kw.arg == 'choices'), None
                )
                assert choices_kw is not None
                break
    assert found_arg, '--overnight_variant argument missing in train.py'

    code_text = _TRAIN_PY.read_text()
    assert 'Bookshelf-Residual-Direct-v0' in code_text

    found_apply_call = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, 'id', None) == 'apply_overnight_variant'
        ):
            found_apply_call = True
            break
    assert found_apply_call, 'apply_overnight_variant call not found in AST'


def test_resume_hyperparams_reapplication():
    """Verify resume branch enforces n_epochs and logs RESUME_HYPERPARAMS."""
    tree = _get_train_tree()

    found_n_epochs_assign = False
    found_resume_print = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == 'n_epochs'
                    and isinstance(target.value, ast.Name)
                    and target.value.id == 'agent'
                ):
                    found_n_epochs_assign = True

        if isinstance(node, ast.Call):
            func_name = getattr(node.func, 'id', None)
            if func_name == 'print':
                for arg in node.args:
                    if isinstance(arg, ast.JoinedStr):
                        for part in arg.values:
                            if (
                                isinstance(part, ast.Constant)
                                and '[RESUME_HYPERPARAMS]' in str(part.value)
                            ):
                                found_resume_print = True
                    elif (
                        isinstance(arg, ast.Constant)
                        and '[RESUME_HYPERPARAMS]' in str(arg.value)
                    ):
                        found_resume_print = True

    assert found_n_epochs_assign, 'agent.n_epochs assignment missing on resume'
    assert found_resume_print, '[RESUME_HYPERPARAMS] print missing in train.py'


def test_finetune_checkpoint_milestones():
    """Verify --finetune_checkpoint_milestones parsing and callback wiring."""
    code_text = _TRAIN_PY.read_text()
    assert '--finetune_checkpoint_milestones' in code_text
    assert 'finetune_checkpoint_milestones requires --resume' in code_text

    tree = _get_train_tree()
    found_milestones_kw = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = getattr(node.func, 'id', None)
            if func_name == 'MilestoneCheckpointCallback':
                for kw in node.keywords:
                    if kw.arg == 'milestones':
                        found_milestones_kw = True

    assert found_milestones_kw, (
        'MilestoneCheckpointCallback milestones missing'
    )


def test_scratch_checkpoint_milestones():
    """Verify --scratch_checkpoint_milestones parsing, wiring, and guards."""
    tree = _get_train_tree()

    # Arg definition exists
    found_arg = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument'
        ):
            if any(
                isinstance(a, ast.Constant)
                and a.value == '--scratch_checkpoint_milestones'
                for a in node.args
            ):
                found_arg = True
    assert found_arg, '--scratch_checkpoint_milestones argument missing'

    # Wired to MilestoneCheckpointCallback with name_tag='fresh'
    # and milestones=
    found_fresh_cb = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, 'id', None) == 'MilestoneCheckpointCallback'
        ):
            has_fresh = any(
                kw.arg == 'name_tag'
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == 'fresh'
                for kw in node.keywords
            )
            has_milestones = any(
                kw.arg == 'milestones'
                and isinstance(kw.value, ast.Name)
                and kw.value.id == 'scratch_milestones'
                for kw in node.keywords
            )
            if has_fresh and has_milestones:
                found_fresh_cb = True
    assert found_fresh_cb, (
        'MilestoneCheckpointCallback(name_tag="fresh", milestones=...) missing'
    )

    # Wiring if branch specifically tests scratch_milestones is not None
    found_wiring_if = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_names = {
                n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
            }
            body_names = {
                n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            }
            if (
                'scratch_milestones' in test_names
                and 'MilestoneCheckpointCallback' in body_names
            ):
                cmp_nodes = [
                    n for n in ast.walk(node.test)
                    if isinstance(n, ast.Compare)
                ]
                has_is_not_none = any(
                    isinstance(c.left, ast.Name)
                    and c.left.id == 'scratch_milestones'
                    and len(c.ops) == 1
                    and isinstance(c.ops[0], ast.IsNot)
                    and len(c.comparators) == 1
                    and isinstance(c.comparators[0], ast.Constant)
                    and c.comparators[0].value is None
                    for c in cmp_nodes
                )
                if has_is_not_none:
                    found_wiring_if = True
    assert found_wiring_if, (
        'Wiring branch must test scratch_milestones is not None'
    )

    # Incompatible with resume and mutually exclusive with interval & finetune
    error_msgs = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'error'
        ):
            for a in node.args:
                if isinstance(a, ast.Constant):
                    error_msgs.append(a.value)

    assert any(
        '--scratch_checkpoint_milestones is incompatible with --resume' in m
        for m in error_msgs
    )
    assert any(
        '--scratch_checkpoint_milestones and '
        '--training_checkpoint_interval_steps are mutually exclusive' in m
        for m in error_msgs
    )
    assert any(
        '--scratch_checkpoint_milestones and '
        '--finetune_checkpoint_milestones are mutually exclusive' in m
        for m in error_msgs
    )
    assert any(
        '--finetune_checkpoint_milestones requires --resume' in m
        for m in error_msgs
    )

    # Plain checkpoint callback guard structurally asserts is None
    guard_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and isinstance(node.test.op, ast.And)
        ):
            test_names = {
                n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
            }
            if {
                'checkpoint_interval',
                'training_checkpoint_interval',
                'finetune_milestones',
                'scratch_milestones',
            }.issubset(test_names):
                body_names = {
                    n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                }
                if (
                    'checkpoint_callback' in body_names
                    or 'vecnormalize_checkpoint_callback' in body_names
                ):
                    guard_node = node
                    break
    assert guard_node is not None, (
        'Plain checkpoint callback guard ast.If not found'
    )

    cmp_nodes = [
        n for n in ast.walk(guard_node.test) if isinstance(n, ast.Compare)
    ]

    scratch_is_none = any(
        isinstance(c.left, ast.Name)
        and c.left.id == 'scratch_milestones'
        and len(c.ops) == 1
        and isinstance(c.ops[0], ast.Is)
        and len(c.comparators) == 1
        and isinstance(c.comparators[0], ast.Constant)
        and c.comparators[0].value is None
        for c in cmp_nodes
    )
    assert scratch_is_none, (
        'Guard must structurally assert scratch_milestones is None'
    )

    finetune_is_none = any(
        isinstance(c.left, ast.Name)
        and c.left.id == 'finetune_milestones'
        and len(c.ops) == 1
        and isinstance(c.ops[0], ast.Is)
        and len(c.comparators) == 1
        and isinstance(c.comparators[0], ast.Constant)
        and c.comparators[0].value is None
        for c in cmp_nodes
    )
    assert finetune_is_none, (
        'Guard must structurally assert finetune_milestones is None'
    )


def test_milestone_callback_structure():
    """Verify MilestoneCheckpointCallback structure."""
    tree = _get_train_tree()
    callback_cls = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == 'MilestoneCheckpointCallback'
        ):
            callback_cls = node
            break
    assert callback_cls is not None, 'MilestoneCheckpointCallback not found'

    init_fn = None
    on_step_fn = None
    for item in callback_cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            init_fn = item
        elif isinstance(item, ast.FunctionDef) and item.name == '_on_step':
            on_step_fn = item

    assert init_fn is not None, '__init__ missing in callback'
    assert on_step_fn is not None, '_on_step missing in callback'

    arg_names = [arg.arg for arg in init_fn.args.args]
    assert 'milestones' in arg_names, 'milestones arg missing in __init__'

    on_step_text = ast.unparse(on_step_fn)
    assert 'milestones' in on_step_text, 'milestones not in _on_step'


def test_no_forbidden_modifications():
    """Verify no forbidden modules and residual env invariants."""
    train_text = _TRAIN_PY.read_text()
    forbidden_names = (
        'bookshelf_robot_deployment',
        'robot_client',
        'executor',
    )
    for forbidden in forbidden_names:
        assert forbidden not in train_text, f'{forbidden} found in train.py'

    # Check AST of bookshelf_residual_env.py does not modify _apply_action,
    # push rewards, or nominal controller in the variant logic.
    env_tree = ast.parse(_ENV_PY.read_text())
    env_cls = None
    for node in ast.walk(env_tree):
        if isinstance(node, ast.ClassDef) and node.name == 'BookshelfEnv':
            env_cls = node
            break
    assert env_cls is not None, 'BookshelfEnv not found in AST'

    # Residual env _apply_action must not touch overnight sweep logic
    apply_action_fn = next(
        item for item in env_cls.body
        if isinstance(item, ast.FunctionDef) and item.name == '_apply_action'
    )
    apply_action_text = ast.unparse(apply_action_fn)
    for marker in (
        'overnight',
        'variant_penalty',
        'insert_wall_margin',
        'insert_lat_penalty_quadratic',
        'insert_action_saturation',
    ):
        assert marker not in apply_action_text, f'{marker} in _apply_action'

    # Check variant penalty block in _get_rewards
    get_rewards_fn = next(
        item for item in env_cls.body
        if isinstance(item, ast.FunctionDef) and item.name == '_get_rewards'
    )
    variant_stmts = []
    in_variant_block = False
    for stmt in get_rewards_fn.body:
        stmt_text = ast.unparse(stmt)
        if 'variant_penalty = torch.zeros_like(rew)' in stmt_text:
            in_variant_block = True
        if in_variant_block:
            variant_stmts.append(stmt)
            if 'rew = rew - insert_mask * variant_penalty' in stmt_text:
                in_variant_block = False

    variant_text = ast.unparse(ast.Module(body=variant_stmts, type_ignores=[]))
    assert 'push_rew' not in variant_text
    assert '_MODE_PUSH' not in variant_text
    assert 'enable_nominal_controller' not in variant_text
