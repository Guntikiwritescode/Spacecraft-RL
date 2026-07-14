"""Reward architectures for the docking environment.

This module encodes a *taxonomy* of reward specifications.  Each class shares
the same interface -- ``reset(initial_distance)`` and
``compute(state, action, info)`` -- so the environment can swap incentive
structures without changing anything else.  Holding the algorithm, dynamics,
seed and time budget fixed while varying only the reward isolates the reward
function as the single independent variable.

Two families are exposed:

**Hackable rewards** -- specifications whose optimum diverges from the intended
objective (safe docking), reproducing historically-documented incentive
failures:

* :class:`HackableReward`  -- the *Cobra Effect* (reward farming).  A
  non-telescoping per-step proximity bonus plus episode-termination-on-success
  makes *hovering near* the target more valuable than *docking*.
* :class:`QuotaHackableReward` -- *Soviet Quota Gaming*.  Rewarding measured
  "activity" (path length / manoeuvre count) makes the agent maximise motion
  rather than achieve the goal, exactly as tonnage/unit quotas produced useless
  output in the planned economy.

**Mitigated rewards** -- specifications engineered so exploitative behaviour no
longer correlates with reward:

* :class:`MitigatedReward`         -- constraint-shaped, sparse-dominant.
* :class:`PotentialBasedReward`    -- Ng et al. (1999) potential-based shaping,
  *provably* policy-invariant, hence exploit-resistant by construction.
* :class:`ConstraintGatedReward`   -- shaping gated on constraint satisfaction.
* :class:`HierarchicalReward`      -- phase-decomposed, first-visit phase
  bonuses defeat phase-oscillation farming.

A note on telescoping (why the *naive* dense reward is a red herring)
---------------------------------------------------------------------
A pure distance-reduction term ``k * (d_{t-1} - d_t)`` telescopes over an
episode to ``k * (d_0 - d_final)``: its cumulative value is path-independent, so
by the potential-based-shaping theorem it *cannot* by itself induce farming.
The exploit in :class:`HackableReward` therefore comes from the additional
*non-telescoping* per-step proximity term -- a pattern that is ubiquitous in
hand-designed dense rewards.  Making this explicit is one of the contributions
of the study.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Coefficient configuration
# --------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    """All tunable reward coefficients in one place.

    Every field is swept-able by the sensitivity study.  Each reward class reads
    only the fields it needs and ignores the rest, so a single config object can
    parameterise the whole taxonomy.
    """

    # Distance shaping (telescoping guidance term). Provides a dense, learnable
    # approach gradient. Because it *telescopes* to k*(d0 - d_final) it is
    # policy-invariant over an episode and cannot by itself create an exploit --
    # so a strong value here aids learning without opening a farming surface.
    shaping_weight: float = 10.0
    shaping_far: float = 3.0
    shaping_close: float = 3.0

    # --- Cobra-Effect exploit surface (non-telescoping "standoff hold" bonus).
    # A per-tick bonus peaked at a *standoff distance* (not at the dock), so the
    # farming optimum is to hover at the standoff ring indefinitely. Because
    # docking (a) pays a bonus smaller than the discounted hold income and
    # (b) *terminates* the episode, ending that income, completing the mission
    # is never worthwhile. This is the crux of the exploit: the reward peaks
    # where the problem *persists*, not where it is solved.
    proximity_weight: float = 3.0
    standoff_distance: float = 20.0
    standoff_width: float = 8.0
    # Station-keeping "engagement bump": a NON-NEGATIVE reward peaked at the
    # standoff. A gentle ramp from far in (reach_far) gives a learnable approach
    # gradient; a steep falloff inside the standoff (reach_near) plus a zero
    # floor in the dock region make closing in strictly worse than holding the
    # ring -- without ever going negative (a negative reward would let the agent
    # "give up" and trigger the divergence terminal to escape it).
    reach_far: float = 120.0
    reach_near: float = 15.0
    # Quota activity is only rewarded inside this radius, and a "leash" penalty
    # for straying beyond it keeps the quota agent orbiting near the target
    # instead of fleeing outward to accrue path length.
    engagement_radius: float = 40.0
    leash_penalty: float = 2.0

    # --- Soviet-quota exploit surface (activity / path-length reward)
    activity_weight: float = 3.0

    # Terminal-objective incentives.
    dock_bonus: float = 5.0            # hackable default: deliberately small
    dock_bonus_mitigated: float = 100.0
    # Terminal-phase per-tick time cost. Set to zero: because the mitigated
    # rewards have no farmable surface, any positive value is unnecessary and
    # forms a barrier that discourages entering the terminal zone at all. Time
    # pressure is instead supplied implicitly by discounting the sparse bonus.
    step_penalty: float = 0.0

    # Constraint penalties.
    velocity_penalty: float = 5.0
    cone_penalty: float = 2.0
    velocity_bonus: float = 0.5        # compliance bonus in the close phase
    gate_closed: float = 0.3           # residual shaping when the safety gate shuts

    # Speed-reduction shaping (telescoping): rewards bleeding off speed as well
    # as distance, so the terminal slow-down needed to dock is guided. Because
    # it telescopes to (v0 - v_final) it is policy-invariant and not farmable.
    speed_shaping_weight: float = 8.0

    # Lateral-offset shaping (telescoping): rewards getting onto the docking
    # axis, so a *safe* (in-cone) dock is reachable. Telescopes to (L0 - L_T).
    # Weighted strongly: at small range even a small lateral offset breaks the
    # approach cone, so nosing onto the +x axis must be well rewarded.
    lateral_shaping_weight: float = 12.0

    # Fuel costs (per unit delta-v used this tick).
    fuel_weight: float = 0.01
    fuel_weight_mitigated: float = 0.05

    # Divergence penalty (applied by the env, listed here for completeness).
    divergence_penalty: float = 50.0

    # Potential-based shaping.
    gamma: float = 0.99
    potential_scale: float = 3.0

    def merged(self, overrides: Optional[Dict[str, Any]]) -> "RewardConfig":
        """Return a copy with ``overrides`` applied (ignoring unknown keys)."""
        if not overrides:
            return self
        known = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return replace(self, **known)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class RewardFunction:
    """Common interface + shared state for all reward architectures."""

    #: Human-readable family tag used in figures/paper.
    family: str = "unknown"

    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()
        self.prev_distance: Optional[float] = None
        self.prev_speed: float = 0.0
        self.prev_lateral: float = 0.0

    def reset(self, initial_distance: float, initial_speed: float = 0.0,
              initial_lateral: float = 0.0) -> None:
        self.prev_distance = float(initial_distance)
        self.prev_speed = float(initial_speed)
        self.prev_lateral = float(initial_lateral)

    def compute(self, state, action, info) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    # Convenience so ``compute`` implementations stay readable.
    def _fuel_term(self, info, weight: float) -> float:
        return -weight * info["fuel_used_this_step"]

    def _standoff_bump(self, d: float) -> float:
        """Non-negative station-keeping bonus, peaked at the standoff distance.

        Gentle linear ramp from far in (slope ~1/reach_far) for a learnable
        approach; steep linear falloff inside the standoff (slope ~1/reach_near)
        with a zero floor, so closing toward the dock is strictly worse than
        holding the ring, and the reward never goes negative.
        """
        cfg = self.cfg
        if d >= cfg.standoff_distance:
            frac = 1.0 - (d - cfg.standoff_distance) / cfg.reach_far
        else:
            frac = 1.0 - (cfg.standoff_distance - d) / cfg.reach_near
        return cfg.proximity_weight * max(0.0, frac)

    def _approach_shaping(self, info, coef: float) -> float:
        """Telescoping distance + speed + lateral-offset reduction shaping
        (policy-invariant, not farmable).

        Rewards getting closer, slowing down, AND getting onto the docking axis,
        guiding the approach, the terminal slow-down, and cone compliance.
        Telescopes to ``coef*[(d0 - d_T) + w_v*(v0 - v_T) + w_L*(L0 - L_T)]``.
        """
        cfg = self.cfg
        d = info["distance_to_target"]
        speed = info["speed"]
        lateral = info["lateral_offset"]
        return coef * ((self.prev_distance - d)
                       + cfg.speed_shaping_weight * (self.prev_speed - speed)
                       + cfg.lateral_shaping_weight * (self.prev_lateral - lateral))

    def _update_prev(self, info) -> None:
        self.prev_distance = info["distance_to_target"]
        self.prev_speed = info["speed"]
        self.prev_lateral = info["lateral_offset"]


# --------------------------------------------------------------------------- #
# HACKABLE: the Cobra Effect (reward farming)
# --------------------------------------------------------------------------- #
class HackableReward(RewardFunction):
    """Station-keeping reward that creates the Cobra Effect.

    ``r = standoff_bump(d) - fuel_weight * fuel_used
          + dock_bonus               (if docked; also ENDS the episode)``

    ``standoff_bump`` is a NON-NEGATIVE station-keeping reward: a gentle ramp up
    from far in (learnable approach), a peak at the nominal standoff distance
    ``d* = standoff_distance``, and a steep falloff to a zero floor inside the
    standoff. It decreases *monotonically* from the standoff toward the dock, and
    is never negative (a negative reward would let the agent "give up" and fly out
    to the divergence terminal to escape it).
    The designer intends "get and stay close" to lead to docking; instead the
    reward-maximising policy PARKS at the standoff ring and never closes in --
    docking would only reduce the per-step reward and, by terminating the
    episode, forfeit all future station-keeping income. The proxy ("maintain
    proximity") is optimised while the objective ("complete the dock") is never
    met -- structurally identical to breeding cobras for a per-head bounty.
    """

    family = "hackable"

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        d = info["distance_to_target"]
        r = self._standoff_bump(d)
        r += self._fuel_term(info, cfg.fuel_weight)
        if info["docked"]:
            r += cfg.dock_bonus
        self.prev_distance = d
        return float(r)


class QuotaHackableReward(RewardFunction):
    """Soviet quota gaming: reward measured *activity*, not achievement.

    ``r = standoff_bump(d)                              # stay engaged near target
          + activity_weight * thrust_magnitude          # "activity" quota (busywork)
          - fuel_weight * fuel_used
          + dock_bonus (if docked)``

    Paying for measured activity -- here control effort / manoeuvres executed --
    rewards *doing things*, not *achieving the goal*. The agent hits its activity
    quota with perpetual make-work manoeuvres at the standoff, burning propellant
    and never settling into the slow, stable approach a dock requires -- the RL
    analogue of a plant maximising output tonnage or unit count with useless
    goods.
    """

    family = "hackable"

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        d = info["distance_to_target"]
        # Quota: reward measured ACTIVITY -- here control effort (thrust /
        # manoeuvres executed) -- while the station-keeping bump keeps the agent
        # near the target. Maximised by constant busywork: the agent burns
        # propellant on perpetual manoeuvres at the standoff, never settling into
        # the slow, stable approach a dock requires. This is the RL analogue of a
        # plant that hits its activity quota with useless make-work output.
        r = self._standoff_bump(d)                                  # stay engaged
        r += cfg.activity_weight * info["thrust_magnitude"]          # busywork
        r += self._fuel_term(info, cfg.fuel_weight)
        if info["docked"]:
            r += cfg.dock_bonus
        self.prev_distance = d
        return float(r)


class ShapedHackableReward(RewardFunction):
    """Sensitivity-study reward: dense docking guidance in tension with a
    tunable standoff-loiter exploit.

    Unlike :class:`HackableReward` (which uses a keep-out penalty to *guarantee*
    non-docking for the benchmark), this reward has full dense guidance toward a
    safe dock -- so docking is reliably learnable -- plus a standoff-loiter bonus
    of tunable strength ``proximity_weight`` and a tunable ``dock_bonus``. The
    reward-maximising behaviour flips from docking to loitering as
    ``proximity_weight`` rises relative to ``dock_bonus``; sweeping the two maps
    the Cobra-Effect emergence boundary.
    """

    family = "hackable"

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        d = info["distance_to_target"]
        speed = info["speed"]
        # Modest dense distance-reduction guidance toward the dock: enough to
        # make docking learnable when the exploit is weak, but weak enough that a
        # strong-enough standoff bump can locally overpower it.
        r = 2.0 * (self.prev_distance - d)
        # Tunable standoff bump (scaled by proximity_weight): a gentle far ramp
        # for the approach, a STEEP near barrier so that above a critical
        # proximity_weight the agent parks at the ring instead of docking.
        if d >= cfg.standoff_distance:
            frac = 1.0 - (d - cfg.standoff_distance) / 60.0
        else:
            frac = 1.0 - (cfg.standoff_distance - d) / 6.0
        r += cfg.proximity_weight * max(0.0, frac)
        if info["docked"]:
            r += cfg.dock_bonus
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)
        self.prev_distance = d
        return float(r)


# --------------------------------------------------------------------------- #
# MITIGATED: constraint-shaped, sparse-dominant
# --------------------------------------------------------------------------- #
class MitigatedReward(RewardFunction):
    """Constraint-shaped reward that eliminates the Cobra Effect.

    Three structural changes remove the exploit:

    1. No per-step proximity income -- there is nothing to farm.
    2. A per-tick penalty in the terminal phase makes *delay* costly.
    3. A dominant sparse ``dock_bonus`` (+100) makes completion overwhelmingly
       preferable to any feasible amount of shaping.

    Constraint-violation penalties (velocity corridor, approach cone) steer a
    safe approach without opening a new exploit surface.
    """

    family = "mitigated"

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        d = info["distance_to_target"]

        # Continuous telescoping distance + speed + lateral shaping in ALL
        # phases -- policy-invariant (not farmable) but always provides a
        # gradient toward a slow, on-axis, close (safely dockable) state.
        coef = cfg.shaping_far if d > 50.0 else cfg.shaping_close
        r = self._approach_shaping(info, coef)
        if info["phase"] == 2:                # terminal phase: time is costly
            r -= cfg.step_penalty
            if info["safe_docked"]:           # safe dock pays big and ends it
                r += cfg.dock_bonus_mitigated

        # Constraint penalties steer a safe approach (no per-step *positive*
        # income that could be farmed).
        if info["velocity_violation"]:
            r -= cfg.velocity_penalty
        if info["cone_violation"]:
            r -= cfg.cone_penalty
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)

        self._update_prev(info)
        return float(r)


# --------------------------------------------------------------------------- #
# MITIGATION ARCHITECTURE 1: potential-based shaping (Ng et al. 1999)
# --------------------------------------------------------------------------- #
class PotentialBasedReward(RewardFunction):
    """Potential-based shaping: provably policy-invariant, exploit-resistant.

    Uses potential ``Phi(s) = -potential_scale * distance`` and shaping
    ``F = gamma * Phi(s') - Phi(s)``.  By the shaping theorem (Ng, Harada &
    Russell, 1999) adding ``F`` to any base reward leaves the optimal policy
    unchanged.  Combined with a sparse safe-docking bonus and constraint
    penalties, the optimum is *dock as fast and safely as possible* -- there is
    no farmable surface, by construction.
    """

    family = "mitigated"

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        # Canonical potential-based shaping F = Phi(s') - Phi(s) with potential
        # Phi(s) = -potential_scale*(distance + w_v*speed + w_L*lateral_offset).
        # This telescopes to Phi(s_T) - Phi(s_0), leaving the optimal policy of
        # the base (sparse safe-dock) reward unchanged (Ng, Harada & Russell
        # 1999). It contains no per-step term that depends on the *level* of the
        # state, so -- unlike a naive gamma*Phi(s')-Phi(s) with unbounded Phi --
        # there is no surface that rewards merely being far or holding position.
        r = self._approach_shaping(info, cfg.potential_scale)
        if info["safe_docked"]:                 # sparse true objective
            r += cfg.dock_bonus_mitigated
        if info["velocity_violation"]:
            r -= cfg.velocity_penalty
        if info["cone_violation"]:
            r -= cfg.cone_penalty
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)
        self._update_prev(info)
        return float(r)


# --------------------------------------------------------------------------- #
# MITIGATION ARCHITECTURE 2: constraint gating
# --------------------------------------------------------------------------- #
class ConstraintGatedReward(RewardFunction):
    """Progress and goal reward are *gated* on safety-constraint satisfaction.

    Two design choices make this a genuine mitigation rather than a leaky one:

    1. There is **no** standalone proximity-hold term -- the farmable surface of
       :class:`HackableReward` is removed.  Dense reward is *telescoping*
       progress only (``prev_d - d``), which is policy-invariant and cannot be
       farmed.
    2. That progress reward is multiplied by a hard safety *gate* (inside the
       velocity corridor AND approach cone), and the large terminal bonus
       requires a *safe* dock.  A per-tick terminal-phase penalty makes even
       constraint-compliant dawdling costly.

    Design note (honest caveat, discussed in the paper): gating *alone* -- i.e.
    gating a dense proximity reward without also removing the hold term -- does
    **not** stop constraint-*compliant* farming.  Gating mitigates only the
    exploits that require a constraint violation; the farmable surface must be
    removed as well.
    """

    family = "mitigated"

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        # Soft gate: unsafe progress earns strongly reduced (not zero) shaping,
        # so a compliant approach is far more rewarding while still leaving a
        # gradient for the agent to *learn* compliance from.
        gate = cfg.gate_closed if (info["velocity_violation"]
                                   or info["cone_violation"]) else 1.0

        r = gate * self._approach_shaping(info, cfg.shaping_close)
        if gate == 0.0:
            r -= (cfg.velocity_penalty + cfg.cone_penalty)
        if info["phase"] == 2:           # terminal phase: time is costly
            r -= cfg.step_penalty
        if info["safe_docked"]:          # goal bonus gated on a *safe* dock
            r += cfg.dock_bonus_mitigated
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)
        self._update_prev(info)
        return float(r)


# --------------------------------------------------------------------------- #
# MITIGATION ARCHITECTURE 3: hierarchical / phase decomposition
# --------------------------------------------------------------------------- #
class HierarchicalReward(RewardFunction):
    """Phase-decomposed reward with *first-visit* phase bonuses.

    Each mission phase (far -> close -> terminal) grants a one-time bonus the
    first time it is reached, tracked via the maximum phase attained.  Because
    re-entering a phase pays nothing, the agent cannot farm reward by
    oscillating across a phase boundary (a Soviet-quota-style exploit).  The
    terminal phase carries a per-tick penalty and a dominant safe-dock bonus.
    """

    family = "mitigated"

    #: one-time bonus for first reaching phase 1 (close) and phase 2 (terminal)
    PHASE_BONUS = {1: 10.0, 2: 25.0}

    def reset(self, initial_distance: float, initial_speed: float = 0.0,
              initial_lateral: float = 0.0) -> None:
        super().reset(initial_distance, initial_speed, initial_lateral)
        self.max_phase = 0

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        phase = info["phase"]
        r = 0.0

        if phase > self.max_phase:
            for p in range(self.max_phase + 1, phase + 1):
                r += self.PHASE_BONUS.get(p, 0.0)
            self.max_phase = phase

        # Gentle within-phase guidance (telescoping, so not farmable).
        r += self._approach_shaping(info, cfg.shaping_close)
        if phase == 2:
            r -= cfg.step_penalty
            if info["safe_docked"]:
                r += cfg.dock_bonus_mitigated
        if info["velocity_violation"]:
            r -= cfg.velocity_penalty
        if info["cone_violation"]:
            r -= cfg.cone_penalty
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)

        self._update_prev(info)
        return float(r)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
REWARD_REGISTRY = {
    "hackable": HackableReward,
    "quota_hackable": QuotaHackableReward,
    "shaped_hackable": ShapedHackableReward,   # sensitivity study only
    "mitigated": MitigatedReward,
    "potential": PotentialBasedReward,
    "constraint_gated": ConstraintGatedReward,
    "hierarchical": HierarchicalReward,
}

#: Which reward types are expected to exhibit hacking vs. mitigate it.
HACKABLE_TYPES = ("hackable", "quota_hackable", "shaped_hackable")
MITIGATED_TYPES = ("mitigated", "potential", "constraint_gated", "hierarchical")


def make_reward(reward_type: str, config: Optional[RewardConfig] = None,
                overrides: Optional[Dict[str, Any]] = None) -> RewardFunction:
    """Factory: build a reward function by name with optional coefficient
    overrides (used by the sensitivity study)."""
    if reward_type not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward_type: {reward_type!r}. "
            f"Choose from {sorted(REWARD_REGISTRY)}."
        )
    cfg = (config or RewardConfig()).merged(overrides)
    return REWARD_REGISTRY[reward_type](cfg)
