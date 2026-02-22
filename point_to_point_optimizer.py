"""Point-to-point trajectory optimizer on a precomputed transition grid.

The implementation follows the dynamic programming specification provided in
`POINT-TO-POINT OPTIMIZER IMPLEMENTATION SPECIFICATION`.

The core idea:
- State at time k is (i, j, l)
- Precomputed horizontal transitions: next_i[k, l, i, j], next_j[k, l, i, j]
- Vertical movement cost: cost_vertical(l_prev, l_new)
- Consumable budget: B

This module exposes a single main entry point:

    optimize_point_to_point(...)

that performs a forward dynamic program using only two time layers of value and
cost, plus backpointer arrays for path reconstruction.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, List

import numpy as np
from tqdm import tqdm


# Type aliases
Index3D = Tuple[int, int, int]
Index4D = Tuple[int, int, int, int]


def default_cost_vertical(l_prev: int, l_new: int) -> float:
    """Default vertical movement cost.

    Uses the absolute level change as cost. This can be replaced by any
    user-provided function with the same signature.
    """

    return float(abs(l_new - l_prev))


def default_reward(
    i: int,
    j: int,
    i2: int,
    j2: int,
    target: Tuple[int, int],
) -> float:
    """Default reward function encouraging motion towards the target.

    Reward is the reduction in Euclidean grid distance to the target between
    the current cell (i, j) and the next cell (i2, j2).
    """

    it, jt = target
    r_prev = np.hypot(i - it, j - jt)
    r_new = np.hypot(i2 - it, j2 - jt)
    return float(r_prev - r_new)


def optimize_point_to_point(
    next_i: np.ndarray,
    next_j: np.ndarray,
    origin: Tuple[int, int],
    target: Tuple[int, int],
    B: float,
    *,
    allowed_levels: Optional[List[int]] = None,
    cost_vertical: Callable[[int, int], float] = default_cost_vertical,
    reward_fn: Callable[[int, int, int, int, Tuple[int, int]], float] = default_reward,
    earliest_arrival: bool = True,
    fixed_arrival_time: Optional[int] = None,
) -> Tuple[bool, Optional[np.ndarray], Optional[float], Optional[float]]:
    """Optimize trajectory from origin to target under a consumable budget.

    Parameters
    ----------
    next_i, next_j : np.ndarray
        Integer arrays of shape ``(Nt, Nz, Nx, Ny)`` giving, for each time step
        ``k``, level ``l`` and horizontal indices ``(i, j)``, the destination
        indices ``(next_i, next_j)`` at time ``k+1``.
    origin : (int, int)
        Origin grid indices ``(i0, j0)`` at time ``k=0``.
    target : (int, int)
        Target grid indices ``(i_target, j_target)``.
    B : float
        Total consumable budget.
    allowed_levels : list of int, optional
        Subset of vertical levels that are allowed. If ``None``, all levels are
        allowed.
    cost_vertical : callable, optional
        Function ``cost_vertical(l_prev, l_new) -> float`` returning the
        consumable cost of moving between vertical levels.
    reward_fn : callable, optional
        Function ``reward_fn(i, j, i2, j2, target) -> float`` returning the
        immediate reward for the horizontal transition.
    earliest_arrival : bool, optional
        If ``True``, stop as soon as the target becomes reachable.
    fixed_arrival_time : int, optional
        If not ``None``, only consider arrivals exactly at this time index
        ``k*`` (0-based). Incompatible with ``earliest_arrival=True``.

    Returns
    -------
    reachable : bool
        ``True`` if the target is reachable within budget, ``False`` otherwise.
    path : np.ndarray or None
        If reachable, an array of shape ``(L, 4)`` with rows
        ``[k, i, j, l]`` describing the optimal path. Otherwise ``None``.
    best_value : float or None
        Objective value of the best path, or ``None`` if unreachable.
    best_cost : float or None
        Total consumable cost of the best path, or ``None`` if unreachable.
    """

    if next_i.shape != next_j.shape:
        raise ValueError("next_i and next_j must have the same shape")

    if fixed_arrival_time is not None and earliest_arrival:
        raise ValueError("earliest_arrival and fixed_arrival_time cannot both be set")

    Nt, Nz, Nx, Ny = next_i.shape
    i0, j0 = origin
    it, jt = target

    if not (0 <= i0 < Nx and 0 <= j0 < Ny):
        raise ValueError("Origin indices out of bounds")
    if not (0 <= it < Nx and 0 <= jt < Ny):
        raise ValueError("Target indices out of bounds")

    # Vertical levels to consider
    all_levels = list(range(Nz))
    if allowed_levels is None:
        levels = all_levels
    else:
        levels = [l for l in allowed_levels if 0 <= l < Nz]
        if not levels:
            raise ValueError("No valid allowed_levels within [0, Nz-1]")

    # Initialize DP arrays (two time layers only)
    best_value_current = np.full((Nx, Ny, Nz), -np.inf, dtype=float)
    best_cost_current = np.full((Nx, Ny, Nz), np.inf, dtype=float)

    # Initialization at k = 0
    for l0 in levels:
        best_value_current[i0, j0, l0] = 0.0
        best_cost_current[i0, j0, l0] = 0.0

    # Backpointer arrays for path reconstruction: prev indices at time k
    prev_i = np.full((Nt, Nx, Ny, Nz), -1, dtype=int)
    prev_j = np.full((Nt, Nx, Ny, Nz), -1, dtype=int)
    prev_l = np.full((Nt, Nx, Ny, Nz), -1, dtype=int)

    # Track best terminal state
    best_overall_value = -np.inf
    best_overall_cost = np.inf
    best_terminal_state: Optional[Index4D] = None  # (k, i, j, l)

    k_start = 0
    k_end = Nt - 1  # last usable k for transitions is Nt-2

    # Progress bar over time steps
    for k in tqdm(range(k_start, k_end), desc="Optimizing trajectory over time"):
        # Prepare next-layer arrays
        best_value_next = np.full((Nx, Ny, Nz), -np.inf, dtype=float)
        best_cost_next = np.full((Nx, Ny, Nz), np.inf, dtype=float)

        # Iterate over all reachable states at time k
        for i in range(Nx):
            for j in range(Ny):
                for l in levels:
                    if not np.isfinite(best_value_current[i, j, l]):
                        continue

                    value_here = best_value_current[i, j, l]
                    cost_here = best_cost_current[i, j, l]

                    # Vertical transitions: restrict to adjacent levels by default
                    for l_new in levels:
                        if abs(l_new - l) > 1:
                            continue

                        delta_cost = cost_vertical(l, l_new)
                        new_cost = cost_here + delta_cost

                        if new_cost > B:
                            continue

                        i2 = int(next_i[k, l_new, i, j])
                        j2 = int(next_j[k, l_new, i, j])

                        reward = reward_fn(i, j, i2, j2, target)
                        new_value = value_here + reward

                        # Dominance check: keep only best (value, cost) pair
                        old_value = best_value_next[i2, j2, l_new]
                        old_cost = best_cost_next[i2, j2, l_new]

                        if not np.isfinite(old_value) or new_value > old_value or (
                            np.isclose(new_value, old_value) and new_cost < old_cost
                        ):
                            best_value_next[i2, j2, l_new] = new_value
                            best_cost_next[i2, j2, l_new] = new_cost

                            # Store backpointer for state at time k+1
                            prev_i[k + 1, i2, j2, l_new] = i
                            prev_j[k + 1, i2, j2, l_new] = j
                            prev_l[k + 1, i2, j2, l_new] = l

        # Move to next time layer
        best_value_current = best_value_next
        best_cost_current = best_cost_next

        # Target check at time k+1
        k_check = k + 1
        if fixed_arrival_time is not None and k_check != fixed_arrival_time:
            continue

        for l in levels:
            val = best_value_current[it, jt, l]
            cost = best_cost_current[it, jt, l]
            if np.isfinite(val):
                if val > best_overall_value or (
                    np.isclose(val, best_overall_value) and cost < best_overall_cost
                ):
                    best_overall_value = val
                    best_overall_cost = cost
                    best_terminal_state = (k_check, it, jt, l)

        if earliest_arrival and best_terminal_state is not None:
            break

    # Infeasible if target never reached
    if best_terminal_state is None:
        return False, None, None, None

    # Reconstruct path
    k_star, i_star, j_star, l_star = best_terminal_state
    path: List[List[int]] = []
    i_curr, j_curr, l_curr = i_star, j_star, l_star

    for t in range(k_star, 0, -1):
        path.append([t, i_curr, j_curr, l_curr])
        i_prev = prev_i[t, i_curr, j_curr, l_curr]
        j_prev = prev_j[t, i_curr, j_curr, l_curr]
        l_prev = prev_l[t, i_curr, j_curr, l_curr]
        if i_prev < 0 or j_prev < 0 or l_prev < 0:
            # Should not happen for a consistent DP table, but guard anyway
            break
        i_curr, j_curr, l_curr = i_prev, j_prev, l_prev

    # Add initial state at t = 0
    path.append([0, i_curr, j_curr, l_curr])
    path.reverse()

    return True, np.array(path, dtype=int), float(best_overall_value), float(best_overall_cost)
