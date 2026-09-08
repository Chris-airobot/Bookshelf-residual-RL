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
