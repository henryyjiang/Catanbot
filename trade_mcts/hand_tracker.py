"""
Partial-observation hand tracker for Catan.

Maintains a set of "particles" — possible hand assignments for all players —
from the perspective of a single observing player. Hands are fully known
except when a rob/steal occurs between two OTHER players (the observer
doesn't see which card was taken).

Observable events that UPDATE hands deterministically:
  - Dice rolls → resource distribution (fully public)
  - Building/buying → resource deductions (fully public)
  - Bank trades → resource swaps (fully public)
  - Player trades → resource swaps (fully public, both sides visible)
  - Monopoly → resource redistribution (fully public)
  - Year of Plenty → resources from bank (fully public)
  - Discards on 7 → cards discarded (public in most implementations)

The ONLY source of uncertainty:
  - Robber steal between two OTHER players: we know WHO robbed WHOM,
    but not WHICH card was taken. This branches the particle set.

Each subsequent observable action prunes impossible particles.
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from data.enums import Resource, LogType


@dataclass
class HandBelief:
    """A single possible hand assignment for one player."""
    resource_counts: dict[int, int] = field(default_factory=lambda: {r.value: 0 for r in Resource})

    def total(self) -> int:
        return sum(self.resource_counts.values())

    def copy(self) -> 'HandBelief':
        return HandBelief(resource_counts=dict(self.resource_counts))

    def has(self, resource: int, amount: int = 1) -> bool:
        return self.resource_counts.get(resource, 0) >= amount

    def add(self, resource: int, amount: int = 1):
        self.resource_counts[resource] = self.resource_counts.get(resource, 0) + amount

    def remove(self, resource: int, amount: int = 1):
        self.resource_counts[resource] = max(0, self.resource_counts.get(resource, 0) - amount)

    def as_list(self) -> list[int]:
        """Return hand as a list of resource type ints (for sampling from)."""
        cards = []
        for res, count in self.resource_counts.items():
            cards.extend([res] * count)
        return cards

    def __eq__(self, other):
        if not isinstance(other, HandBelief):
            return False
        return self.resource_counts == other.resource_counts

    def __hash__(self):
        return hash(tuple(sorted(self.resource_counts.items())))


@dataclass
class Particle:
    """One possible world state — a complete hand assignment for all players."""
    hands: dict[int, HandBelief]  # player_color → HandBelief
    weight: float = 1.0

    def copy(self) -> 'Particle':
        return Particle(
            hands={c: h.copy() for c, h in self.hands.items()},
            weight=self.weight,
        )


class HandTracker:
    """
    Tracks possible hand states from the perspective of one player.

    For the observer's own hand, there is always exactly one possibility
    (they know their own cards). For other players, the particle set
    branches only when an unobserved steal occurs.
    """

    MAX_PARTICLES = 200  # cap to keep computation bounded

    def __init__(self, observer_color: int, player_colors: list[int]):
        self.observer_color = observer_color
        self.player_colors = player_colors

        # Start with a single particle — all hands known at game start
        initial_hands = {c: HandBelief() for c in player_colors}
        self.particles: list[Particle] = [Particle(hands=initial_hands)]

    @property
    def num_particles(self) -> int:
        return len(self.particles)

    @property
    def uncertainty_level(self) -> float:
        """0.0 = perfect knowledge, 1.0 = maximum uncertainty."""
        if len(self.particles) <= 1:
            return 0.0
        # Measure as fraction of max particles
        return min(len(self.particles) / self.MAX_PARTICLES, 1.0)

    # ─── Deterministic updates (applied to ALL particles identically) ───

    def observe_resource_gain(self, player_color: int, resource: int, amount: int = 1):
        """Player gains resources from dice roll, year of plenty, etc."""
        for particle in self.particles:
            particle.hands[player_color].add(resource, amount)

    def observe_resource_loss(self, player_color: int, resource: int, amount: int = 1):
        """Player loses resources from building, bank trade, etc."""
        for particle in self.particles:
            particle.hands[player_color].remove(resource, amount)

    def observe_set_hand(self, player_color: int, resource_cards: list[int]):
        """Directly set a player's hand (used when we have full info, e.g. from game state)."""
        counts = {r.value: 0 for r in Resource}
        for card in resource_cards:
            counts[card] = counts.get(card, 0) + 1
        for particle in self.particles:
            particle.hands[player_color] = HandBelief(resource_counts=dict(counts))

    def observe_trade(
        self,
        player_a: int, gives_a: dict[int, int],
        player_b: int, gives_b: dict[int, int],
    ):
        """A trade between two players — fully public."""
        for particle in self.particles:
            for res, amt in gives_a.items():
                particle.hands[player_a].remove(res, amt)
                particle.hands[player_b].add(res, amt)
            for res, amt in gives_b.items():
                particle.hands[player_b].remove(res, amt)
                particle.hands[player_a].add(res, amt)

    def observe_bank_trade(
        self, player_color: int, gives: dict[int, int], receives: dict[int, int]
    ):
        """Player trades with the bank — fully public."""
        for particle in self.particles:
            for res, amt in gives.items():
                particle.hands[player_color].remove(res, amt)
            for res, amt in receives.items():
                particle.hands[player_color].add(res, amt)

    def observe_monopoly(self, player_color: int, resource: int, total_gained: int):
        """Monopoly card played — all of one resource type goes to player.
        We know the total gained because it's announced."""
        for particle in self.particles:
            for other_color in self.player_colors:
                if other_color != player_color:
                    amt = particle.hands[other_color].resource_counts.get(resource, 0)
                    particle.hands[other_color].resource_counts[resource] = 0
            particle.hands[player_color].add(resource, total_gained)

    def observe_discard(self, player_color: int, discarded: dict[int, int]):
        """Player discards on a 7 — public if the cards are shown."""
        for particle in self.particles:
            for res, amt in discarded.items():
                particle.hands[player_color].remove(res, amt)

    # ─── The key uncertainty event ───

    def observe_steal(
        self,
        thief_color: int,
        victim_color: int,
        stolen_resource: Optional[int] = None,
    ):
        """
        A player steals one card from another via robber.

        If the observer is the thief or the victim, stolen_resource is known.
        If the observer is neither, stolen_resource is None and we branch.
        """
        if stolen_resource is not None:
            # Observer was involved — deterministic update
            for particle in self.particles:
                particle.hands[victim_color].remove(stolen_resource, 1)
                particle.hands[thief_color].add(stolen_resource, 1)
            return

        # Observer was NOT involved — branch on all possible stolen cards
        new_particles = []
        for particle in self.particles:
            victim_hand = particle.hands[victim_color]
            victim_cards = victim_hand.as_list()

            if not victim_cards:
                # Victim had no cards — no steal actually happens
                new_particles.append(particle)
                continue

            # Get unique resource types the victim could have
            possible_steals = Counter(victim_cards)

            for resource, count in possible_steals.items():
                new_particle = particle.copy()
                # Weight by probability of stealing this resource type
                new_particle.weight = particle.weight * (count / len(victim_cards))
                new_particle.hands[victim_color].remove(resource, 1)
                new_particle.hands[thief_color].add(resource, 1)
                new_particles.append(new_particle)

        self.particles = new_particles
        self._prune_and_resample()

    # ─── Pruning and consistency ───

    def prune_inconsistent(self, player_color: int, known_min: dict[int, int]):
        """
        Remove particles where a player can't have the minimum resources
        we know they must have (e.g., they just built something that requires
        specific resources, so they must have had at least those).
        """
        self.particles = [
            p for p in self.particles
            if all(
                p.hands[player_color].resource_counts.get(res, 0) >= amt
                for res, amt in known_min.items()
            )
        ]
        # If we pruned everything, something went wrong — reset to uniform
        if not self.particles:
            self._reset_to_uniform()

    def _prune_and_resample(self):
        """Keep particle count bounded by resampling weighted."""
        if len(self.particles) <= self.MAX_PARTICLES:
            return

        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight <= 0:
            self.particles = self.particles[:self.MAX_PARTICLES]
            return

        for p in self.particles:
            p.weight /= total_weight

        # Deduplicate identical particles (merge weights)
        unique: dict[tuple, Particle] = {}
        for p in self.particles:
            # Create a hashable key from all hands
            key = tuple(
                (c, tuple(sorted(p.hands[c].resource_counts.items())))
                for c in sorted(p.hands.keys())
            )
            if key in unique:
                unique[key].weight += p.weight
            else:
                unique[key] = p

        self.particles = list(unique.values())

        # If still too many, resample by weight
        if len(self.particles) > self.MAX_PARTICLES:
            weights = [p.weight for p in self.particles]
            total = sum(weights)
            probs = [w / total for w in weights]
            indices = random.choices(
                range(len(self.particles)),
                weights=probs,
                k=self.MAX_PARTICLES,
            )
            seen = set()
            resampled = []
            for idx in indices:
                if idx not in seen:
                    resampled.append(self.particles[idx])
                    seen.add(idx)
            self.particles = resampled

        # Reset weights to uniform after resampling
        for p in self.particles:
            p.weight = 1.0 / len(self.particles)

    def _reset_to_uniform(self):
        """Fallback if pruning eliminates all particles."""
        # This shouldn't happen in normal play, but just in case
        initial_hands = {c: HandBelief() for c in self.player_colors}
        self.particles = [Particle(hands=initial_hands)]

    # ─── Sampling for MCTS ───

    def sample_hand(self, player_color: int) -> dict[int, int]:
        """
        Sample a plausible hand for a player from the belief distribution.
        Returns resource_counts dict.
        """
        if len(self.particles) == 1:
            return dict(self.particles[0].hands[player_color].resource_counts)

        weights = [p.weight for p in self.particles]
        total = sum(weights)
        probs = [w / total for w in weights]
        particle = random.choices(self.particles, weights=probs, k=1)[0]
        return dict(particle.hands[player_color].resource_counts)

    def sample_all_hands(self) -> dict[int, dict[int, int]]:
        """Sample a consistent set of hands for all players."""
        if len(self.particles) == 1:
            return {
                c: dict(self.particles[0].hands[c].resource_counts)
                for c in self.player_colors
            }

        weights = [p.weight for p in self.particles]
        total = sum(weights)
        probs = [w / total for w in weights]
        particle = random.choices(self.particles, weights=probs, k=1)[0]
        return {
            c: dict(particle.hands[c].resource_counts)
            for c in self.player_colors
        }

    def get_most_likely_hand(self, player_color: int) -> dict[int, int]:
        """Return the most probable hand for a player."""
        if len(self.particles) == 1:
            return dict(self.particles[0].hands[player_color].resource_counts)

        best = max(self.particles, key=lambda p: p.weight)
        return dict(best.hands[player_color].resource_counts)

    def get_hand_distribution(self, player_color: int) -> dict[int, tuple[float, float]]:
        """
        Return mean and std of each resource count for a player
        across all particles. Useful for the trade models as soft features.
        """
        import numpy as np

        counts_per_resource = {r.value: [] for r in Resource}
        weights = [p.weight for p in self.particles]
        total_w = sum(weights)

        for p in self.particles:
            hand = p.hands[player_color]
            for r in Resource:
                counts_per_resource[r.value].append(
                    hand.resource_counts.get(r.value, 0)
                )

        result = {}
        for res, counts in counts_per_resource.items():
            arr = np.array(counts)
            w = np.array(weights) / total_w
            mean = np.average(arr, weights=w)
            var = np.average((arr - mean) ** 2, weights=w)
            result[res] = (float(mean), float(var ** 0.5))

        return result
