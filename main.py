import heapq
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys

class Graph:
    """Graph representation using adjacency list"""
    def __init__(self):
        self.vertices = set()
        self.edges = defaultdict(list)

    def add_edge(self, u, v, weight):
        """Add weighted edge between vertices u and v"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges[u].append((v, weight))
        self.edges[v].append((u, weight))  # For undirected graph

    def get_neighbors(self, vertex):
        """Get neighbors of a vertex"""
        return self.edges[vertex]

    def get_vertices(self):
        """Get all vertices"""
        return list(self.vertices)

class FibonacciHeapNode:
    """Node for Fibonacci Heap"""
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.degree = 0
        self.marked = False
        self.parent = None
        self.child = None
        self.left = self
        self.right = self

class FibonacciHeap:
    """Fibonacci Heap implementation for Dijkstra's algorithm"""
    def __init__(self):
        self.min_node = None
        self.total_nodes = 0
        self.node_map = {}  # Maps values to nodes for decrease_key operation

    def insert(self, key, value):
        """Insert a new node with given key and value"""
        new_node = FibonacciHeapNode(key, value)
        self.node_map[value] = new_node

        if self.min_node is None:
            self.min_node = new_node
        else:
            self._add_to_root_list(new_node)
            if new_node.key < self.min_node.key:
                self.min_node = new_node

        self.total_nodes += 1
        return new_node

    def extract_min(self):
        """Extract the minimum node"""
        if self.min_node is None:
            return None

        min_node = self.min_node

        # Add all children to root list
        if min_node.child:
            child = min_node.child
            while True:
                next_child = child.right
                child.parent = None
                self._add_to_root_list(child)
                child = next_child
                if child == min_node.child:
                    break

        # Remove min_node from root list
        self._remove_from_root_list(min_node)

        if min_node == min_node.right:
            self.min_node = None
        else:
            self.min_node = min_node.right
            self._consolidate()

        self.total_nodes -= 1
        if min_node.value in self.node_map:
            del self.node_map[min_node.value]

        return (min_node.key, min_node.value)

    def decrease_key(self, value, new_key):
        """Decrease the key of a node"""
        if value not in self.node_map:
            return False

        node = self.node_map[value]
        if new_key > node.key:
            return False

        node.key = new_key
        parent = node.parent

        if parent and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)

        if node.key < self.min_node.key:
            self.min_node = node

        return True

    def is_empty(self):
        """Check if heap is empty"""
        return self.min_node is None

    def _add_to_root_list(self, node):
        """Add node to root list"""
        if self.min_node is None:
            self.min_node = node
            node.left = node.right = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node

    def _remove_from_root_list(self, node):
        """Remove node from root list"""
        if node.right == node:
            return
        node.left.right = node.right
        node.right.left = node.left

    def _consolidate(self):
        """Consolidate the heap"""
        max_degree = int(np.log2(self.total_nodes)) + 1
        degree_table = [None] * (max_degree + 1)

        # Collect all root nodes
        root_nodes = []
        if self.min_node:
            current = self.min_node
            while True:
                root_nodes.append(current)
                current = current.right
                if current == self.min_node:
                    break

        # Consolidate nodes with same degree
        for node in root_nodes:
            degree = node.degree
            while degree_table[degree] is not None:
                other = degree_table[degree]
                if node.key > other.key:
                    node, other = other, node
                self._link(other, node)
                degree_table[degree] = None
                degree += 1
            degree_table[degree] = node

        # Find new minimum
        self.min_node = None
        for node in degree_table:
            if node is not None:
                if self.min_node is None or node.key < self.min_node.key:
                    self.min_node = node

    def _link(self, child, parent):
        """Link child to parent"""
        self._remove_from_root_list(child)
        child.parent = parent

        if parent.child is None:
            parent.child = child
            child.left = child.right = child
        else:
            child.left = parent.child
            child.right = parent.child.right
            parent.child.right.left = child
            parent.child.right = child

        parent.degree += 1
        child.marked = False

    def _cut(self, node, parent):
        """Cut node from parent"""
        if parent.child == node:
            if node.right == node:
                parent.child = None
            else:
                parent.child = node.right

        node.left.right = node.right
        node.right.left = node.left
        parent.degree -= 1

        self._add_to_root_list(node)
        node.parent = None
        node.marked = False

    def _cascading_cut(self, node):
        """Cascading cut operation"""
        parent = node.parent
        if parent:
            if not node.marked:
                node.marked = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)

class DijkstraImplementations:
    """Different implementations of Dijkstra's algorithm"""

    @staticmethod
    def naive_dijkstra(graph, start):
        """Naive implementation - O(V²)"""
        distances = {vertex: float('inf') for vertex in graph.get_vertices()}
        distances[start] = 0
        visited = set()

        while len(visited) < len(graph.get_vertices()):
            # Find unvisited vertex with minimum distance
            current = None
            min_dist = float('inf')

            for vertex in graph.get_vertices():
                if vertex not in visited and distances[vertex] < min_dist:
                    min_dist = distances[vertex]
                    current = vertex

            if current is None:
                break

            visited.add(current)

            # Update distances to neighbors
            for neighbor, weight in graph.get_neighbors(current):
                if neighbor not in visited:
                    new_dist = distances[current] + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist

        return distances

    @staticmethod
    def binary_heap_dijkstra(graph, start):
        """Binary heap implementation - O((V + E) log V)"""
        distances = {vertex: float('inf') for vertex in graph.get_vertices()}
        distances[start] = 0

        # Priority queue: (distance, vertex)
        pq = [(0, start)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            # Update distances to neighbors
            for neighbor, weight in graph.get_neighbors(current):
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))

        return distances

    @staticmethod
    def fibonacci_heap_dijkstra(graph, start):
        """Fibonacci heap implementation - O(E + V log V)"""
        distances = {vertex: float('inf') for vertex in graph.get_vertices()}
        distances[start] = 0

        fib_heap = FibonacciHeap()
        visited = set()

        # Insert all vertices into fibonacci heap
        for vertex in graph.get_vertices():
            fib_heap.insert(distances[vertex], vertex)

        while not fib_heap.is_empty():
            current_dist, current = fib_heap.extract_min()

            if current in visited:
                continue

            visited.add(current)

            # Update distances to neighbors
            for neighbor, weight in graph.get_neighbors(current):
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        fib_heap.decrease_key(neighbor, new_dist)

        return distances

class GraphGenerator:
    """Generate different types of graphs for testing"""

    @staticmethod
    def generate_random_graph(num_vertices, num_edges, max_weight=100):
        """Generate random connected graph"""
        graph = Graph()
        vertices = list(range(num_vertices))

        # Ensure connectivity by creating a spanning tree
        for i in range(1, num_vertices):
            u = random.randint(0, i-1)
            v = i
            weight = random.randint(1, max_weight)
            graph.add_edge(u, v, weight)

        # Add remaining edges randomly
        edges_added = num_vertices - 1
        while edges_added < num_edges:
            u = random.randint(0, num_vertices-1)
            v = random.randint(0, num_vertices-1)
            if u != v:
                weight = random.randint(1, max_weight)
                graph.add_edge(u, v, weight)
                edges_added += 1

        return graph

    @staticmethod
    def generate_dense_graph(num_vertices, density=0.8, max_weight=100):
        """Generate dense graph"""
        max_edges = num_vertices * (num_vertices - 1) // 2
        num_edges = int(max_edges * density)
        return GraphGenerator.generate_random_graph(num_vertices, num_edges, max_weight)

    @staticmethod
    def generate_sparse_graph(num_vertices, max_weight=100):
        """Generate sparse graph"""
        num_edges = num_vertices * 2  # Sparse: roughly 2E = V
        return GraphGenerator.generate_random_graph(num_vertices, num_edges, max_weight)

class PerformanceAnalyzer:
    """Analyze performance of different Dijkstra implementations"""

    @staticmethod
    def measure_execution_time(func, *args):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        return end_time - start_time, result

    @staticmethod
    def analyze_implementations(graph_sizes, graph_types=['sparse', 'dense']):
        """Analyze all implementations across different graph sizes"""
        results = {
            'sizes': graph_sizes,
            'naive': {'sparse': [], 'dense': []},
            'binary_heap': {'sparse': [], 'dense': []},
            'fibonacci_heap': {'sparse': [], 'dense': []}
        }

        for size in graph_sizes:
            print(f"Testing graph size: {size}")

            for graph_type in graph_types:
                print(f"  Graph type: {graph_type}")

                # Generate graph
                if graph_type == 'sparse':
                    graph = GraphGenerator.generate_sparse_graph(size)
                else:
                    graph = GraphGenerator.generate_dense_graph(size)

                start_vertex = 0

                # Test naive implementation (skip for large graphs)
                if size <= 1000:  # Naive is too slow for large graphs
                    time_naive, _ = PerformanceAnalyzer.measure_execution_time(
                        DijkstraImplementations.naive_dijkstra, graph, start_vertex
                    )
                    results['naive'][graph_type].append(time_naive)
                    print(f"    Naive: {time_naive:.4f}s")
                else:
                    results['naive'][graph_type].append(None)

                # Test binary heap implementation
                time_binary, _ = PerformanceAnalyzer.measure_execution_time(
                    DijkstraImplementations.binary_heap_dijkstra, graph, start_vertex
                )
                results['binary_heap'][graph_type].append(time_binary)
                print(f"    Binary Heap: {time_binary:.4f}s")

                # Test fibonacci heap implementation
                time_fib, _ = PerformanceAnalyzer.measure_execution_time(
                    DijkstraImplementations.fibonacci_heap_dijkstra, graph, start_vertex
                )
                results['fibonacci_heap'][graph_type].append(time_fib)
                print(f"    Fibonacci Heap: {time_fib:.4f}s")

        return results

    @staticmethod
    def plot_results(results):
        """Plot performance comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        graph_types = ['sparse', 'dense']
        colors = {'naive': 'red', 'binary_heap': 'blue', 'fibonacci_heap': 'green'}

        for i, graph_type in enumerate(graph_types):
            ax = axes[i]
            sizes = results['sizes']

            for impl_name, color in colors.items():
                times = results[impl_name][graph_type]
                # Filter out None values for plotting
                valid_data = [(s, t) for s, t in zip(sizes, times) if t is not None]
                if valid_data:
                    valid_sizes, valid_times = zip(*valid_data)
                    ax.plot(valid_sizes, valid_times, 'o-', color=color, 
                           label=impl_name.replace('_', ' ').title(), linewidth=2, markersize=6)

            ax.set_xlabel('Number of Vertices')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title(f'Performance Comparison - {graph_type.title()} Graphs')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('dijkstra_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

class TestSuite:
    """Test suite to verify correctness of implementations"""

    @staticmethod
    def verify_correctness():
        """Verify all implementations produce same results"""
        print("Testing correctness of implementations...")

        # Create test graph
        graph = Graph()
        edges = [
            (0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5),
            (2, 3, 8), (2, 4, 10), (3, 4, 2)
        ]

        for u, v, w in edges:
            graph.add_edge(u, v, w)

        start = 0

        # Run all implementations
        result_naive = DijkstraImplementations.naive_dijkstra(graph, start)
        result_binary = DijkstraImplementations.binary_heap_dijkstra(graph, start)
        result_fib = DijkstraImplementations.fibonacci_heap_dijkstra(graph, start)

        print("Results:")
        print(f"Naive:          {result_naive}")
        print(f"Binary Heap:    {result_binary}")
        print(f"Fibonacci Heap: {result_fib}")

        # Verify results are identical
        if result_naive == result_binary == result_fib:
            print("✅ All implementations produce identical results!")
            return True
        else:
            print("❌ Results differ between implementations!")
            return False

    @staticmethod
    def test_edge_cases():
        """Test edge cases"""
        print("\nTesting edge cases...")

        # Single vertex graph
        graph = Graph()
        graph.vertices.add(0)

        result = DijkstraImplementations.binary_heap_dijkstra(graph, 0)
        assert result[0] == 0, "Single vertex test failed"
        print("✅ Single vertex test passed")

        # Disconnected graph
        graph = Graph()
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)

        result = DijkstraImplementations.binary_heap_dijkstra(graph, 0)
        assert result[0] == 0 and result[1] == 1, "Connected component test failed"
        assert result[2] == float('inf') and result[3] == float('inf'), "Disconnected component test failed"
        print("✅ Disconnected graph test passed")

def main():
    """Main function to run the complete analysis"""
    print("=== Dijkstra's Algorithm Performance Analysis ===\n")

    # Test correctness first
    if not TestSuite.verify_correctness():
        print("Correctness test failed! Exiting...")
        return

    TestSuite.test_edge_cases()

    print("\n" + "="*50)
    print("Starting Performance Analysis...")
    print("="*50)

    # Define test parameters
    graph_sizes = [50, 100, 200, 500, 1000, 2000]

    # Run analysis
    results = PerformanceAnalyzer.analyze_implementations(graph_sizes)

    # Plot results
    print("\nGenerating performance plots...")
    PerformanceAnalyzer.plot_results(results)

    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*50)

    print("\nTime Complexity Analysis:")
    print("1. Naive Implementation:     O(V²)")
    print("2. Binary Heap:             O((V + E) log V)")
    print("3. Fibonacci Heap:          O(E + V log V)")

    print("\nKey Observations:")
    print("- Fibonacci heap shows theoretical advantage for dense graphs")
    print("- Binary heap performs well in practice due to better constant factors")
    print("- Naive implementation becomes impractical for large graphs")

    print("\nFiles generated:")
    print("- dijkstra_performance_comparison.png")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()