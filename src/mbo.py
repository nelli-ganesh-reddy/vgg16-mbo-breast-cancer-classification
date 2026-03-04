import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def run_mbo(X, y, max_iter=2, num_butterflies=30):

    def fitness(solution):
        if np.sum(solution) == 0:
            return 0
        X_sel = X[:, solution == 1]
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(clf, X_sel, y, cv=3, scoring='accuracy')
        return np.mean(scores)

    num_features = X.shape[1]
    migration_ratio = 0.7

    population = np.random.randint(0, 2, (num_butterflies, num_features))
    fitness_scores = np.array([fitness(ind) for ind in population])

    for t in range(max_iter):

        idx = np.argsort(-fitness_scores)
        population = population[idx]
        fitness_scores = fitness_scores[idx]

        land1_size = int(migration_ratio * num_butterflies)
        land1 = population[:land1_size]
        land2 = population[land1_size:]

        new_population = []

        for i in range(land1_size):
            donor = land1[np.random.randint(0, land1_size)]
            butterfly = population[i].copy()
            mask = np.random.rand(num_features) < 0.1
            butterfly[mask] = donor[mask]
            new_population.append(butterfly)

        for i in range(len(land2)):
            butterfly = land2[i].copy()
            mask = np.random.rand(num_features) < 0.05
            butterfly[mask] = 1 - butterfly[mask]
            new_population.append(butterfly)

        population = np.array(new_population)
        fitness_scores = np.array([fitness(ind) for ind in population])

        print(f"Iteration {t+1}/{max_iter} | Best accuracy = {np.max(fitness_scores):.4f}")

    best_solution = population[np.argmax(fitness_scores)]
    selected = np.where(best_solution == 1)[0]

    return selected