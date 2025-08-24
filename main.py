# Copyright (©) 2025, Alexander Suvorov. All rights reserved.
import numpy as np
import math
import time
import random
from typing import List, Tuple
from numba import njit
import argparse


@njit(fastmath=True, cache=True)
def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


class ExactTSPSolver:
    def __init__(self, num_points: int, seed: int = 42):
        self.num_points = num_points
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.points = self.generate_random_points()
        self.dist_matrix = compute_distance_matrix(self.points)
        self.total_permutations = self.calculate_total_permutations()

        self.best_path = []
        self.best_distance = float('inf')
        self.visited_count = 0
        self.mst_cache = {}
        self.start_time = 0
        self.last_update = 0

        self.nearest_neighbors = self.precompute_nearest_neighbors()
        self.greedy_path = []
        self.greedy_distance = float('inf')
        self.final_visited_count = 0

    def generate_random_points(self) -> np.ndarray:
        points = np.zeros((self.num_points, 2), dtype=np.float64)
        for i in range(self.num_points):
            points[i, 0] = random.random() * 1000
            points[i, 1] = random.random() * 1000
        return points

    def calculate_total_permutations(self) -> int:
        if self.num_points <= 1:
            return 1
        if self.num_points > 20:
            return 2 ** 63 - 1
        result = 1
        for i in range(2, self.num_points):
            result *= i
        return result

    def precompute_nearest_neighbors(self) -> List[List[int]]:
        neighbors = []
        for i in range(self.num_points):
            other_points = [(j, self.dist_matrix[i, j]) for j in range(self.num_points) if j != i]
            other_points.sort(key=lambda x: x[1])
            k = min(10, len(other_points))
            neighbors.append([x[0] for x in other_points[:k]])
        return neighbors

    def calculate_path_distance(self, path: List[int]) -> float:
        total = 0.0
        n = len(path)
        for i in range(n):
            j = (i + 1) % n
            total += self.dist_matrix[path[i], path[j]]
        return total

    def calculate_partial_distance(self, path: List[int]) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            total += self.dist_matrix[path[i], path[i + 1]]
        return total

    def two_opt(self, path: List[int]) -> Tuple[List[int], float]:
        if len(path) < 4:
            return path, self.calculate_path_distance(path)

        current_path = path.copy()
        current_distance = self.calculate_path_distance(current_path)
        n = len(current_path)
        improved = True
        iteration = 0
        max_iterations = 1000

        while improved and iteration < max_iterations:
            iteration += 1
            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1

            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    delta = (-self.dist_matrix[current_path[i - 1], current_path[i]] -
                             self.dist_matrix[current_path[j], current_path[j + 1]] +
                             self.dist_matrix[current_path[i - 1], current_path[j]] +
                             self.dist_matrix[current_path[i], current_path[j + 1]])

                    if delta < best_delta - 1e-10:
                        best_delta = delta
                        best_i, best_j = i, j
                        improved = True

            if improved:
                current_path[best_i:best_j + 1] = current_path[best_i:best_j + 1][::-1]
                current_distance += best_delta

        return current_path, current_distance

    def multi_start_greedy(self) -> Tuple[List[int], float]:
        best_path = []
        best_dist = float('inf')

        for start_point in range(min(5, self.num_points)):
            path = [start_point]
            unvisited = set(range(self.num_points))
            unvisited.remove(start_point)

            while unvisited:
                current = path[-1]
                min_dist = float('inf')
                next_point = -1

                for neighbor in self.nearest_neighbors[current]:
                    if neighbor in unvisited:
                        dist = self.dist_matrix[current, neighbor]
                        if dist < min_dist:
                            min_dist = dist
                            next_point = neighbor

                if next_point == -1:
                    for point in unvisited:
                        dist = self.dist_matrix[current, point]
                        if dist < min_dist:
                            min_dist = dist
                            next_point = point

                path.append(next_point)
                unvisited.remove(next_point)

            optimized_path, optimized_dist = self.two_opt(path)

            if optimized_dist < best_dist:
                best_dist = optimized_dist
                best_path = optimized_path

        return best_path, best_dist

    def calculate_mst_kruskal(self, points: List[int]) -> float:
        if len(points) <= 1:
            return 0.0

        sorted_points = sorted(points)
        key = tuple(sorted_points)
        if key in self.mst_cache:
            return self.mst_cache[key]

        edges = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                from_idx, to_idx = points[i], points[j]
                edges.append((self.dist_matrix[from_idx, to_idx], from_idx, to_idx))

        edges.sort(key=lambda x: x[0])

        point_to_index = {point: idx for idx, point in enumerate(points)}

        uf = UnionFind(len(points))
        mst_length = 0.0
        edges_used = 0

        for weight, from_point, to_point in edges:
            if edges_used == len(points) - 1:
                break

            idx_from = point_to_index[from_point]
            idx_to = point_to_index[to_point]

            if uf.find(idx_from) != uf.find(idx_to):
                uf.union(idx_from, idx_to)
                mst_length += weight
                edges_used += 1

        self.mst_cache[key] = mst_length
        return mst_length

    def min_connection_to_path(self, unvisited: List[int], path: List[int]) -> float:
        if not unvisited:
            return 0.0

        start, end = path[0], path[-1]
        min_start = min(self.dist_matrix[start, point] for point in unvisited)
        min_end = min(self.dist_matrix[end, point] for point in unvisited)

        return min_start + min_end

    def calculate_lower_bound(self, current_path: List[int], visited: List[bool]) -> float:
        current_length = self.calculate_partial_distance(current_path)

        unvisited = [i for i in range(self.num_points) if not visited[i]]

        if not unvisited:
            return current_length + self.dist_matrix[current_path[-1], current_path[0]]

        mst_length = self.calculate_mst_kruskal(unvisited)
        connection_length = self.min_connection_to_path(unvisited, current_path)

        return current_length + mst_length + connection_length

    def format_large_number(self, n: int) -> str:
        if n < 1000:
            return str(n)
        elif n < 1000000:
            return f"{n / 1000:.1f} thousand."
        elif n < 1000000000:
            return f"{n / 1000000:.1f} million."
        elif n < 1000000000000:
            return f"{n / 1000000000:.1f} billion."
        else:
            return "so many"

    def format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def print_progress(self):
        elapsed = time.time() - self.start_time
        paths_per_sec = self.visited_count / elapsed if elapsed > 0 else 0
        print(f"\rChecked: {self.format_large_number(self.visited_count)} paths | "
              f"Speed: {paths_per_sec:.0f}/sec | Time: {self.format_time(elapsed)}", end="")

    def branch_and_bound_recursive(self, current_path: List[int], current_distance: float,
                                   visited: List[bool], threshold: float):
        lower_bound = self.calculate_lower_bound(current_path, visited)
        if lower_bound >= threshold - 1e-10:
            return

        if len(current_path) == self.num_points:
            final_distance = current_distance + self.dist_matrix[current_path[-1], current_path[0]]
            self.visited_count += 1

            if time.time() - self.last_update > 0.5:
                self.print_progress()
                self.last_update = time.time()

            if final_distance < threshold - 1e-10:
                self.best_distance = final_distance
                self.best_path = current_path.copy()
            return

        last_point = current_path[-1]
        points_to_try = []

        for neighbor in self.nearest_neighbors[last_point]:
            if not visited[neighbor]:
                points_to_try.append(neighbor)

        for i in range(self.num_points):
            if not visited[i] and i not in points_to_try:
                points_to_try.append(i)

        points_to_try.sort(key=lambda x: self.dist_matrix[last_point, x])

        for next_point in points_to_try:
            new_distance = current_distance + self.dist_matrix[last_point, next_point]

            if new_distance >= threshold - 1e-10:
                continue

            visited[next_point] = True
            new_path = current_path + [next_point]

            self.branch_and_bound_recursive(new_path, new_distance, visited, threshold)
            visited[next_point] = False

    def adaptive_search(self) -> Tuple[List[int], float, int]:
        print("1. Launching the multi-start greedy algorithm...")
        self.greedy_path, self.greedy_distance = self.multi_start_greedy()
        print(f"   ✅ Multi-start greedy + 2-opt: length = {self.greedy_distance:.2f}")

        current_threshold = self.greedy_distance * 0.90
        step = 0.07
        best_path = self.greedy_path
        best_dist = self.greedy_distance
        found_any = False

        print(f"   🎯 We start the search from {current_threshold:.2f} (90.0%)")

        for iteration in range(200):
            self.best_path = []
            self.best_distance = float('inf')
            self.visited_count = 0
            self.mst_cache = {}

            visited = [False] * self.num_points
            visited[0] = True

            print(
                f"   🔍 Threshold: {current_threshold:.2f} ({current_threshold / self.greedy_distance * 100:.1f}%)... ",
                end="", flush=True)

            start_time = time.time()
            self.branch_and_bound_recursive([0], 0.0, visited, current_threshold)
            search_time = time.time() - start_time

            if self.best_distance < current_threshold:
                print(f"✓ found: {self.best_distance:.2f} ({self.format_time(search_time)})")
                best_path = self.best_path.copy()
                best_dist = self.best_distance
                found_any = True
                current_threshold = best_dist * (1.0 - step)
                self.final_visited_count = self.visited_count
            else:
                print(f"✗ cut off ({self.format_time(search_time)})")

                if found_any:
                    print(f"   🏆 The optimum has been found: {best_dist:.2f}")
                    return best_path, best_dist, self.final_visited_count
                else:
                    current_threshold = current_threshold * (1.0 + step)

                    if current_threshold >= self.greedy_distance:
                        print("   ⚠️  No better solutions than greedy found")
                        return self.greedy_path, self.greedy_distance, self.visited_count

        print(f"   🏆 Best found: {best_dist:.2f}")
        return best_path, best_dist, self.final_visited_count

    def solve(self) -> Tuple[List[int], float, float, int]:
        self.start_time = time.time()
        self.last_update = self.start_time

        optimal_path, optimal_dist, visited_count = self.adaptive_search()
        elapsed = time.time() - self.start_time

        return optimal_path, optimal_dist, elapsed, visited_count


def main():
    parser = argparse.ArgumentParser(description='TSP Solver')
    parser.add_argument('-n', '--num-points', type=int, default=10, help='Number of points')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random generation')
    args = parser.parse_args()

    if args.num_points < 3:
        print("Error: Minimum 3 dots required")
        return

    if args.num_points > 25:
        total_perms = 1
        for i in range(2, args.num_points):
            total_perms *= i

        def format_large_number(n):
            if n < 1000:
                return str(n)
            elif n < 1000000:
                return f"{n / 1000:.1f} thousand."
            elif n < 1000000000:
                return f"{n / 1000000:.1f} million."
            else:
                return f"{n / 1000000000:.1f} billion."

        print(
            f"⚠️  WARNING: for {args.num_points} points there will be approximately {format_large_number(total_perms)} permutations")
        print("This may take a considerable amount of time.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled by user")
            return

    solver = ExactTSPSolver(args.num_points, args.seed)

    print("==================================================")
    print(f"🚀 TSP SOLVER (ORACLE v2) - {args.num_points} POINTS")
    print(f"🔢 SEED: {args.seed}")
    print("==================================================")

    print("\n📍 Coordinates of points:")
    for i, point in enumerate(solver.points):
        print(f"   Dot {i}: ({point[0]:.2f}, {point[1]:.2f})")

    optimal_path, optimal_dist, elapsed, visited_count = solver.solve()

    print("\n📊 RESULTS:")
    print("==================================================")
    print(f"Number of points: {args.num_points}")
    print(f"Seed: {args.seed}")
    print(f"Total possible paths: {solver.format_large_number(solver.total_permutations)}")
    print(f"Checked paths: {solver.format_large_number(visited_count)}")
    print(f"Execution time: {elapsed:.2f} seconds")
    if elapsed > 0:
        print(f"Speed: {visited_count / elapsed:.0f} paths/sec")
    print(f"Greedy + 2-opt: {solver.greedy_distance:.6f}")
    print(f"Optimal length: {optimal_dist:.6f}")
    improvement = solver.greedy_distance - optimal_dist
    improvement_percent = (improvement / solver.greedy_distance) * 100
    print(f"Improvement: {improvement:.6f} ({improvement_percent:.3f}%)")

    print(f"\nGreedy way: {solver.greedy_path}")
    print(f"The optimal path: {optimal_path}")

    filename = f"tsp_result_n{args.num_points}_seed{args.seed}.txt"
    try:
        with open(filename, 'w') as f:
            f.write(f"SEED: {args.seed}\n")
            f.write("Points:\n")
            for i, p in enumerate(solver.points):
                f.write(f"{i}: ({p[0]:.6f}, {p[1]:.6f})\n")
            f.write(f"Greedy + 2-opt: {solver.greedy_distance:.6f}\n")
            f.write(f"Optimal: {optimal_dist:.6f}\n")
            f.write(f"Improvement: {improvement:.6f} ({improvement_percent:.3f}%)\n")
            f.write(f"Greedy path: {solver.greedy_path}\n")
            f.write(f"Optimal path: {optimal_path}\n")
            f.write(f"Time: {elapsed:.2f} seconds\n")
            f.write(f"Paths checked: {visited_count}\n")
            f.write(f"Total paths: {solver.total_permutations}\n")
        print(f"\n💾 The results are saved in {filename}")
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")


if __name__ == "__main__":
    main()
