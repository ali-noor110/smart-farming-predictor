import numpy as np
import random

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FEATURE_BOUNDS = {
    "N":           (0, 140),
    "P":           (5, 145),
    "K":           (5, 205),
    "temperature": (5, 45),
    "humidity":    (10, 100),
    "ph":          (3.5, 9.5),
    "rainfall":    (20, 300),
}

CROP_CONDITIONS = {
    "rice":        [80, 40, 40, 23, 82, 6.5, 200],
    "maize":       [80, 45, 40, 22, 65, 6.0, 75],
    "chickpea":    [45, 70, 80, 18, 17, 7.3, 75],
    "kidneybeans": [20, 65, 20, 20, 22, 5.7, 105],
    "pigeonpeas":  [20, 65, 20, 28, 49, 5.8, 145],
    "mothbeans":   [21, 48, 20, 28, 53, 6.9, 50],
    "mungbean":    [20, 47, 20, 28, 85, 6.7, 45],
    "blackgram":   [40, 67, 19, 30, 66, 7.1, 67],
    "lentil":      [18, 65, 19, 24, 65, 6.9, 45],
    "pomegranate": [18, 18, 40, 22, 90, 6.0, 105],
    "banana":      [100, 82, 50, 27, 80, 6.0, 105],
    "mango":       [20, 27, 30, 31, 50, 6.3, 95],
    "grapes":      [23, 133, 200, 24, 82, 6.5, 65],
    "watermelon":  [99, 17, 50, 25, 85, 6.5, 50],
    "muskmelon":   [100, 17, 50, 28, 92, 6.5, 25],
    "apple":       [20, 134, 199, 22, 92, 6.0, 112],
    "orange":      [20, 16, 10, 23, 92, 7.0, 110],
    "papaya":      [49, 59, 50, 34, 92, 6.7, 145],
    "coconut":     [23, 17, 30, 27, 94, 6.5, 150],
    "cotton":      [118, 46, 43, 24, 80, 6.7, 80],
    "jute":        [78, 46, 40, 25, 80, 6.5, 175],
    "coffee":      [101, 28, 29, 25, 58, 6.5, 155],
}


def _random_chromosome():
    bounds = list(FEATURE_BOUNDS.values())
    return [random.uniform(lo, hi) for lo, hi in bounds]


def _fitness(chromosome, target_crop):
    """Fitness = negative normalized Euclidean distance to ideal (maximized)."""
    ideal = CROP_CONDITIONS[target_crop]
    bounds = list(FEATURE_BOUNDS.values())
    dist = 0.0
    for g, i, (lo, hi) in zip(chromosome, ideal, bounds):
        dist += ((g - i) / (hi - lo + 1e-9)) ** 2
    return -np.sqrt(dist)   # negative because we maximize


def _tournament_select(population, fitnesses, k=3):
    contestants = random.sample(range(len(population)), k)
    best = max(contestants, key=lambda i: fitnesses[i])
    return population[best][:]


def _crossover(p1, p2, rate=0.8):
    if random.random() < rate:
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1[:], p2[:]


def _mutate(chromosome, rate=0.05):
    bounds = list(FEATURE_BOUNDS.values())
    for i in range(len(chromosome)):
        if random.random() < rate:
            lo, hi = bounds[i]
            chromosome[i] = random.uniform(lo, hi)
    return chromosome


def _clamp(chromosome):
    bounds = list(FEATURE_BOUNDS.values())
    return [max(lo, min(hi, g)) for g, (lo, hi) in zip(chromosome, bounds)]


def run_ga(target_crop: str,
           pop_size: int = 100,
           generations: int = 200,
           crossover_rate: float = 0.8,
           mutation_rate: float = 0.05,
           seed: int = 42) -> dict:

    if generations < 1:
        raise ValueError("generations must be at least 1")
    if pop_size < 2:
        raise ValueError("pop_size must be at least 2")
    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate must be between 0 and 1")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if target_crop not in CROP_CONDITIONS:
        raise ValueError(f"Unknown crop: {target_crop}")

    random.seed(seed)
    np.random.seed(seed)

    population = [_random_chromosome() for _ in range(pop_size)]
    fitness_history = []
    best_chromosome = None
    best_fitness = -np.inf

    for gen in range(generations):
        fitnesses = [_fitness(c, target_crop) for c in population]
        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_chromosome = population[gen_best_idx][:]

        fitness_history.append(gen_best_fit)

        next_gen = [best_chromosome[:]]
        while len(next_gen) < pop_size:
            p1 = _tournament_select(population, fitnesses)
            p2 = _tournament_select(population, fitnesses)
            c1, c2 = _crossover(p1, p2, crossover_rate)
            next_gen.append(_clamp(_mutate(c1, mutation_rate)))
            if len(next_gen) < pop_size:
                next_gen.append(_clamp(_mutate(c2, mutation_rate)))
        population = next_gen

    ideal = CROP_CONDITIONS[target_crop]
    result = {
        "best_chromosome": {f: round(v, 2) for f, v in zip(FEATURE_NAMES, best_chromosome)},
        "best_fitness":    round(best_fitness, 4),
        "fitness_history": fitness_history,
        "ideal_conditions": {f: v for f, v in zip(FEATURE_NAMES, ideal)},
    }
    return result


def run_ga_experiment(target_crop: str,
                      seeds: list = None,
                      generations: int = 200) -> dict:
    seeds = [42, 43, 44, 45, 46] if seeds is None else seeds
    runs = [
        run_ga(
            target_crop,
            pop_size=100,
            generations=generations,
            crossover_rate=0.8,
            mutation_rate=0.05,
            seed=seed,
        )
        for seed in seeds
    ]
    fitnesses = [run["best_fitness"] for run in runs]
    return {
        "seeds": seeds,
        "runs": runs,
        "mean_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
    }
