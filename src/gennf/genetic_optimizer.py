"""Real-coded genetic algorithm (maximises fitness).

Defaults mirror the MATLAB script where it states them: population 100, 50 generations,
mutation rate 0.05, parents drawn from the top half, single-point crossover. Differences:
elitism is added (the MATLAB code discarded the best individuals each generation), mutation is
per-gene Gaussian, and fitness never references the ground-truth target (see docs/methodology.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GAConfig:
    population_size: int = 200
    generations: int = 100
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.05
    elite: int = 5
    tournament: int = 5
    crossover_rate: float = 0.9
    mutation_decay: float = 0.995


@dataclass
class GAResult:
    best: np.ndarray
    best_fitness: float
    history: list[float] = field(default_factory=list)


def genetic_optimize(fitness_fn, initial_population: np.ndarray, cfg: GAConfig,
                     rng: np.random.Generator) -> GAResult:
    pop = np.array(initial_population, dtype=float)
    if len(pop) != cfg.population_size:
        raise ValueError("initial_population size must equal cfg.population_size")
    dim = pop.shape[1]; history = []
    sigma = cfg.mutation_sigma
    for _ in range(cfg.generations):
        fit = np.array([fitness_fn(ind) for ind in pop])
        order = np.argsort(-fit); pop, fit = pop[order], fit[order]
        history.append(float(fit[0]))
        new = [pop[i].copy() for i in range(cfg.elite)]
        while len(new) < cfg.population_size:
            # tournament selection
            def tour():
                idx = rng.choice(len(pop), size=cfg.tournament, replace=False)
                return pop[idx[np.argmax(fit[idx])]].copy()
            p1, p2 = tour(), tour()
            if rng.random() < cfg.crossover_rate:
                alpha = rng.random(dim)
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()
            mask = rng.random(dim) < cfg.mutation_rate
            child = child + mask * rng.normal(0, sigma, size=dim)
            new.append(child)
        pop = np.array(new)
        sigma *= cfg.mutation_decay
    fit = np.array([fitness_fn(ind) for ind in pop]); b = int(np.argmax(fit))
    history.append(float(fit[b]))
    return GAResult(best=pop[b].copy(), best_fitness=float(fit[b]), history=history)
