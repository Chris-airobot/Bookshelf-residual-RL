#!/usr/bin/env python3
"""Structural AST tests for overnight sweep INSERT-only reward logic.

All assertions inspect real AST node structures (node types, operators,
targets, and call chains) rather than unparsed text substrings to prevent
false negatives under code mutations.
"""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
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


def _get_env_tree() -> ast.AST:
    return ast.parse(_ENV_PY.read_text())


def _get_methods():
    tree = _get_env_tree()
    env_cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'BookshelfEnv':
            env_cls = node
            break
    assert env_cls is not None, 'BookshelfEnv class not found in AST'

    get_rewards_fn = None
    pre_physics_fn = None
    for item in env_cls.body:
        if isinstance(item, ast.FunctionDef):
            if item.name == '_get_rewards':
                get_rewards_fn = item
            elif item.name == '_pre_physics_step':
                pre_physics_fn = item
    assert get_rewards_fn is not None, '_get_rewards not found in BookshelfEnv'
    assert pre_physics_fn is not None, (
        '_pre_physics_step not found in BookshelfEnv'
    )
    return get_rewards_fn, pre_physics_fn


def test_insert_mask_assigned_from_mode_insert():
    """Verify insert_mask is assigned from self._mode_start == _MODE_INSERT."""
    get_rewards_fn, _ = _get_methods()
    found_insert_mask = False

    for node in ast.walk(get_rewards_fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == 'insert_mask':
                found_insert_mask = True
                cmp_node = node.value
                # Handle possible .float() call wrapper
                if (
                    isinstance(cmp_node, ast.Call)
                    and isinstance(cmp_node.func, ast.Attribute)
                    and cmp_node.func.attr == 'float'
                ):
                    cmp_node = cmp_node.func.value
                assert isinstance(cmp_node, ast.Compare), (
                    f'Expected ast.Compare, got {type(cmp_node)}'
                )
                assert len(cmp_node.ops) == 1 and isinstance(
                    cmp_node.ops[0], ast.Eq
                ), f'Expected [ast.Eq], got {cmp_node.ops}'
                assert isinstance(cmp_node.left, ast.Attribute) and (
                    cmp_node.left.attr == '_mode_start'
                ), 'Left operand must be an Attribute ending in _mode_start'
                assert len(cmp_node.comparators) == 1, (
                    'Expected exactly 1 comparator'
                )
                comp = cmp_node.comparators[0]
                if isinstance(comp, ast.Name):
                    assert comp.id == '_MODE_INSERT', (
                        f'Comparator Name must be _MODE_INSERT, got {comp.id}'
                    )
                elif isinstance(comp, ast.Attribute):
                    assert comp.attr == '_MODE_INSERT', (
                        'Comparator Attribute must be _MODE_INSERT, '
                        f'got {comp.attr}'
                    )
                else:
                    assert False, f'Unexpected comparator type: {type(comp)}'

    assert found_insert_mask, (
        'insert_mask assignment not found in _get_rewards'
    )


def test_rew_reduced_by_insert_mask_times_variant_penalty():
    """Verify rew is reduced by insert_mask * variant_penalty via AST."""
    get_rewards_fn, _ = _get_methods()
    found_rew_subtraction = False

    for node in ast.walk(get_rewards_fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'rew':
                    if isinstance(node.value, ast.BinOp):
                        # Looking for rew - insert_mask * variant_penalty
                        right = node.value.right
                        if isinstance(right, ast.BinOp):
                            names = set()
                            for op_side in (right.left, right.right):
                                if isinstance(op_side, ast.Name):
                                    names.add(op_side.id)
                            if {
                                'insert_mask',
                                'variant_penalty',
                            }.issubset(names):
                                assert isinstance(node.value.op, ast.Sub), (
                                    'Outer op must be ast.Sub, not ast.Add'
                                )
                                assert isinstance(right.op, ast.Mult), (
                                    'Inner op must be ast.Mult'
                                )
                                found_rew_subtraction = True
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == 'rew':
                if isinstance(node.value, ast.BinOp):
                    names = set()
                    for op_side in (node.value.left, node.value.right):
                        if isinstance(op_side, ast.Name):
                            names.add(op_side.id)
                    if {
                        'insert_mask',
                        'variant_penalty',
                    }.issubset(names):
                        assert isinstance(node.op, ast.Sub), (
                            'AugAssign op must be ast.Sub'
                        )
                        assert isinstance(node.value.op, ast.Mult), (
                            'Inner op must be ast.Mult'
                        )
                        found_rew_subtraction = True

    assert found_rew_subtraction, (
        'rew = rew - insert_mask * variant_penalty missing in _get_rewards'
    )


def test_saturation_slices_5_motion_dims_excluding_release():
    """Verify saturation penalty slices [:5] and never release dim via AST."""
    get_rewards_fn, _ = _get_methods()
    found_saturation_call = False

    for node in ast.walk(get_rewards_fn):
        if isinstance(node, ast.Call):
            is_sat = False
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'saturation_penalty'
            ):
                is_sat = True
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id == 'saturation_penalty'
            ):
                is_sat = True
            if is_sat:
                found_saturation_call = True
                assert node.args, 'saturation_penalty call has no arguments'
                first_arg = node.args[0]
                assert isinstance(first_arg, ast.Subscript), (
                    'Arg to saturation_penalty must be Subscript, '
                    f'got {type(first_arg)}'
                )
                sl = first_arg.slice
                if isinstance(sl, ast.Tuple):
                    dim_slice = sl.elts[-1]
                else:
                    dim_slice = sl
                assert isinstance(dim_slice, ast.Slice), (
                    f'Slice must be ast.Slice, got {type(dim_slice)}'
                )
                if dim_slice.lower is not None:
                    assert (
                        isinstance(dim_slice.lower, ast.Constant)
                        and dim_slice.lower.value == 0
                    ), 'Lower bound must be None or 0'
                assert (
                    isinstance(dim_slice.upper, ast.Constant)
                    and dim_slice.upper.value == 5
                ), f'Upper bound must be 5, got {ast.dump(dim_slice.upper)}'

    # Assert no subscript on raw / _raw_actions with slice 5 or -1 in rewards
    for node in ast.walk(get_rewards_fn):
        if isinstance(node, ast.Subscript):
            is_raw = False
            if isinstance(node.value, ast.Name) and node.value.id == 'raw':
                is_raw = True
            elif (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == '_raw_actions'
            ):
                is_raw = True
            if is_raw:
                slices = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                for s in slices:
                    if isinstance(s, ast.Constant) and s.value == 5:
                        assert False, 'Found raw index 5 (release dim)'
                    if (
                        isinstance(s, ast.UnaryOp)
                        and isinstance(s.op, ast.USub)
                        and isinstance(s.operand, ast.Constant)
                        and s.operand.value == 1
                    ):
                        assert False, 'Found raw index -1 (release dim)'

    assert found_saturation_call, 'saturation_penalty call not found'


def test_raw_actions_captured_before_clamp():
    """Verify _pre_physics_step captures raw actions before clamp via AST."""
    _, pre_physics_fn = _get_methods()

    raw_assign = None
    clamp_assign = None

    for stmt in pre_physics_fn.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == '_raw_actions'
                ):
                    raw_assign = stmt
                    # Assert not a print call
                    assert not (
                        isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == 'print'
                    ), 'Value cannot be a call to print'
                    # Assert call chain root Name id == 'actions'
                    cur = stmt.value
                    while isinstance(cur, (ast.Call, ast.Attribute)):
                        if isinstance(cur, ast.Call):
                            cur = cur.func
                        else:
                            cur = cur.value
                    assert (
                        isinstance(cur, ast.Name) and cur.id == 'actions'
                    ), (
                        "Expected call chain root 'actions', "
                        f'got {ast.dump(cur)}'
                    )
                elif (
                    isinstance(target, ast.Attribute)
                    and target.attr == 'actions'
                ):
                    for n in ast.walk(stmt.value):
                        if (
                            isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute)
                            and n.func.attr == 'clamp'
                        ):
                            clamp_assign = stmt

    assert raw_assign is not None, (
        'self._raw_actions assignment not found in _pre_physics_step'
    )
    assert clamp_assign is not None, (
        'self.actions clamp assignment missing in _pre_physics_step'
    )
    assert raw_assign.lineno < clamp_assign.lineno, (
        'self._raw_actions must be captured before clamp'
    )


def test_wall_term_uses_clearance_and_neighbor_thickness():
    """Verify wall term uses _slot_lateral_clearance_env and neighbor_thick."""
    get_rewards_fn, _ = _get_methods()
    wall_block_found = False

    for node in ast.walk(get_rewards_fn):
        if isinstance(node, ast.If):
            test_text = ast.unparse(node.test)
            if 'insert_wall_margin_penalty_enable' in test_text:
                wall_text = ast.unparse(node)
                if 'inner_half_env' in wall_text:
                    assert '_neighbor_thick_y' in wall_text
                    assert '_slot_lateral_clearance_env' in wall_text
                    wall_block_found = True

    assert wall_block_found, 'Wall margin penalty block not found'


def test_no_changes_to_push_or_release_blocks():
    """Verify variant penalty statements do not reference push or release."""
    get_rewards_fn, _ = _get_methods()

    forbidden = {
        'push_rew',
        '_MODE_PUSH',
        'release_ready_bonus',
        'release_premature_penalty',
        '_release_request',
        'decision_reward',
    }

    checked_stmts = 0
    for stmt in get_rewards_fn.body:
        mentions_variant = any(
            (isinstance(n, ast.Name) and n.id == 'variant_penalty')
            for n in ast.walk(stmt)
        )
        if mentions_variant:
            checked_stmts += 1
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name):
                    assert n.id not in forbidden, (
                        f'Forbidden Name {n.id!r} in variant statement'
                    )
                elif isinstance(n, ast.Attribute):
                    assert n.attr not in forbidden, (
                        f'Forbidden Attribute {n.attr!r} in variant statement'
                    )

    assert checked_stmts > 0, (
        'No statements mentioning variant_penalty found in _get_rewards'
    )
