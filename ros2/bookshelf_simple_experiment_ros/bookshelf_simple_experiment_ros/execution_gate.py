"""Shared fail-closed command gate for the simple physical experiment."""


def hardware_commands_allowed(requested_execution, shadow_full_sequence):
    """Return true only when execution is requested and shadow is disabled."""

    return bool(requested_execution) and not bool(shadow_full_sequence)
