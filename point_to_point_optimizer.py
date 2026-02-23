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
    start_levels: Optional[List[int]] = None,
    target_levels: Optional[List[int]] = None,
    cost_vertical: Callable[[int, int], float] = default_cost_vertical,
    reward_fn: Callable[[int, int, int, int, Tuple[int, int]], float] = default_reward,
    earliest_arrival: bool = True,
    fixed_arrival_time: Optional[int] = None,
    k_best: int = 1,
) -> Tuple[bool, Optional[object], Optional[object], Optional[object]]:
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
    start_levels : list of int, optional
        Levels at which the trajectory is allowed to start at the origin. If
        ``None``, all ``allowed_levels`` are used.
    target_levels : list of int, optional
        Levels at which arrival at the target is considered valid. If ``None``,
        all ``allowed_levels`` are used.
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

    if k_best < 1:
        raise ValueError("k_best must be at least 1")
    if k_best > 1 and earliest_arrival:
        raise ValueError("Set earliest_arrival=False when requesting k_best > 1")

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

    # Start and target level subsets (must be within levels)
    if start_levels is None:
        start_levels_eff = levels
    else:
        start_levels_eff = [l for l in start_levels if l in levels]
        if not start_levels_eff:
            raise ValueError("No valid start_levels within allowed_levels")

    if target_levels is None:
        target_levels_eff = levels
    else:
        target_levels_eff = [l for l in target_levels if l in levels]
        if not target_levels_eff:
            raise ValueError("No valid target_levels within allowed_levels")

    # Initialize DP arrays (two time layers only)
    best_value_current = np.full((Nx, Ny, Nz), -np.inf, dtype=float)
    best_cost_current = np.full((Nx, Ny, Nz), np.inf, dtype=float)

    # Initialization at k = 0
    for l0 in start_levels_eff:
        best_value_current[i0, j0, l0] = 0.0
        best_cost_current[i0, j0, l0] = 0.0

    # Backpointer arrays for path reconstruction: prev indices at time k
    prev_i = np.full((Nt, Nx, Ny, Nz), -1, dtype=int)
    prev_j = np.full((Nt, Nx, Ny, Nz), -1, dtype=int)
    prev_l = np.full((Nt, Nx, Ny, Nz), -1, dtype=int)

    # Track all terminal states at the target for later ranking
    terminal_states: List[Tuple[float, float, Index4D]] = []  # (value, cost, (k,i,j,l))

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

        for l in target_levels_eff:
            val = best_value_current[it, jt, l]
            cost = best_cost_current[it, jt, l]
            if np.isfinite(val):
                terminal_states.append((float(val), float(cost), (k_check, it, jt, l)))

    # Infeasible if target never reached
    if not terminal_states:
        return False, None, None, None

    # Sort terminal states by (value desc, cost asc) and keep top-k
    terminal_states.sort(key=lambda x: (-x[0], x[1]))
    selected = terminal_states[:k_best]

    paths: List[np.ndarray] = []
    values: List[float] = []
    costs: List[float] = []

    for val, cost, (k_star, i_star, j_star, l_star) in selected:
        path_steps: List[List[int]] = []
        i_curr, j_curr, l_curr = i_star, j_star, l_star

        for t in range(k_star, 0, -1):
            path_steps.append([t, i_curr, j_curr, l_curr])
            i_prev = prev_i[t, i_curr, j_curr, l_curr]
            j_prev = prev_j[t, i_curr, j_curr, l_curr]
            l_prev = prev_l[t, i_curr, j_curr, l_curr]
            if i_prev < 0 or j_prev < 0 or l_prev < 0:
                break
            i_curr, j_curr, l_curr = i_prev, j_prev, l_prev

        path_steps.append([0, i_curr, j_curr, l_curr])
        path_steps.reverse()

        paths.append(np.array(path_steps, dtype=int))
        values.append(val)
        costs.append(cost)

    # Backwards compatibility: if k_best == 1, return single path and scalars
    if k_best == 1:
        return True, paths[0], values[0], costs[0]

    return True, paths, np.array(values), np.array(costs)
