import math
import random

def read_input_file(filename="input.txt"):
    """
    Reads the input file and returns a list of city coordinates [(x, y, z), ...].
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    N = int(lines[0])
    cities = []
    for i in range(1, N+1):
        x, y, z = map(int, lines[i].split())
        cities.append((x, y, z))
    return cities

def write_output_file(distance, best_path, filename="output.txt"):
    """
    Writes the output file in the required format:
      - First line: the total distance traveled
      - Next N+1 lines: each line has 3 integers (x, y, z)
    """
    with open(filename, "w") as f:
        f.write(f"{distance:.3f}\n")
        for (x, y, z) in best_path:
            f.write(f"{x} {y} {z}\n")

def create_distance_matrix(cities):
    """
    Precompute the 3D Euclidean distances in an NxN matrix.
    """
    n = len(cities)
    distances = [[0.0]*n for _ in range(n)]
    for i in range(n):
        x1, y1, z1 = cities[i]
        for j in range(i+1, n):
            x2, y2, z2 = cities[j]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            distances[i][j] = dist
            distances[j][i] = dist
    return distances

def total_route_distance(route, distances):
    """Compute total traveling distance (including return to start)."""
    distance_sum = 0.0
    n = len(route)
    for i in range(n):
        current_city = route[i]
        next_city = route[(i + 1) % n]
        distance_sum += distances[current_city][next_city]
    return distance_sum

def create_initial_population(pop_size, num_cities):
    """Creates a list of random permutations (routes)."""
    population = []
    base_route = list(range(num_cities))
    for _ in range(pop_size):
        route = base_route[:]
        random.shuffle(route)
        population.append(route)
    return population

def two_opt(route, distances, max_iterations=50):
    """2-opt local search."""
    best_distance = total_route_distance(route, distances)
    improved = True
    iteration = 0
    n = len(route)
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                new_route = route[:]
                new_route[i+1:j+1] = reversed(new_route[i+1:j+1])
                new_dist = total_route_distance(new_route, distances)
                if new_dist < best_distance:
                    route = new_route
                    best_distance = new_dist
                    improved = True
                    break
            if improved:
                break
    return route

def tournament_selection(population, distances, tournament_size=5, elite_size=2):
    """Tournament selection with elitism."""
    pop_with_dist = [(route, total_route_distance(route, distances)) for route in population]
    pop_with_dist.sort(key=lambda x: x[1])
    mating_pool = [pop_with_dist[i][0] for i in range(elite_size)]
    while len(mating_pool) < len(population):
        candidates = random.sample(pop_with_dist, tournament_size)
        candidates.sort(key=lambda x: x[1])
        mating_pool.append(candidates[0][0])
    return mating_pool

def order_crossover(parent1, parent2):
    """Order Crossover (OX) for TSP."""
    size = len(parent1)
    child = [None] * size
    start = random.randint(0, size - 2)
    end = random.randint(start, size - 1)
    for i in range(start, end+1):
        child[i] = parent1[i]
    p2_index = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_index] in child:
                p2_index += 1
            child[i] = parent2[p2_index]
            p2_index += 1
    return child

def breed_population(mating_pool, distances):
    """Crossover and 2-opt refinement."""
    children = []
    pool_size = len(mating_pool)
    random.shuffle(mating_pool)
    for i in range(0, pool_size, 2):
        parent1 = mating_pool[i]
        parent2 = mating_pool[(i+1) % pool_size]
        child1 = two_opt(order_crossover(parent1, parent2), distances, max_iterations=20)
        child2 = two_opt(order_crossover(parent2, parent1), distances, max_iterations=20)
        children.extend([child1, child2])
    return children[:pool_size]

def mutate(route, mutation_rate=0.02):
    """Swap mutation."""
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route)-1)
            route[i], route[j] = route[j], route[i]
    return route

def mutate_population(population, mutation_rate=0.02):
    for i in range(len(population)):
        population[i] = mutate(population[i], mutation_rate)
    return population

def genetic_algorithm(cities, pop_size=100, elite_size=2, tournament_size=5, mutation_rate=0.02, generations=200):
    """Main GA loop for 3D TSP."""
    distances = create_distance_matrix(cities)
    num_cities = len(cities)
    population = create_initial_population(pop_size, num_cities)
    best_route, best_dist = None, float("inf")
    for gen in range(generations):
        mating_pool = tournament_selection(population, distances, tournament_size, elite_size)
        children = breed_population(mating_pool, distances)
        population = mutate_population(children, mutation_rate)
        for route in population:
            dist = total_route_distance(route, distances)
            if dist < best_dist:
                best_dist, best_route = dist, route[:]
    return best_route, best_dist, distances

def main():
    cities = read_input_file("input.txt")
    best_route, best_distance, _ = genetic_algorithm(cities, pop_size=100, elite_size=5)
    best_path_coords = [cities[idx] for idx in best_route]
    best_path_coords.append(cities[best_route[0]])
    write_output_file(best_distance, best_path_coords, filename="output.txt")

if __name__ == "__main__":
    main()
