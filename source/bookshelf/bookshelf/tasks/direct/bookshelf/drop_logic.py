"""Shared drop classification for the bookshelf task phases."""

from __future__ import annotations

import torch


def book_dropped_mask(
    *,
    lowest_z: torch.Tensor,
    on_shelf: torch.Tensor,
    mode: torch.Tensor,
    insert_mode: int,
    true_ground_z: float,
    shelf_drop_z: float,
) -> torch.Tensor:
    """Classify drops without treating a low, grasped book as released."""
    threshold = torch.where(
        mode == int(insert_mode),
        torch.full_like(lowest_z, float(true_ground_z)),
        torch.full_like(lowest_z, float(shelf_drop_z)),
    )
    return (lowest_z <= threshold) & ~on_shelf
