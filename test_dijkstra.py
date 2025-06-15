
#!/usr/bin/env python3
"""
Comprehensive Test Suite for Dijkstra's Algorithm Implementations
Author: Kabore Taryam William Rodrigue
GitHub: https://github.com/JIMMY62m24
"""

import unittest
import sys
import os
import time
from unittest.mock import patch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    Graph, DijkstraImplementations, GraphGenerator, 
    FibonacciHeap, TestSuite, PerformanceAnalyzer
)

class TestDijkstraImplementations(unittest.TestCase):
    """Comprehensive test suite for Dijkstra algorithm implementations"""

    def setUp(self):
        """Set up test fixtures"""
        self.graph = Graph()
        # Create a standard test graph
        edges = [
            (0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5),
            (2, 3, 8), (2, 4, 10), (3, 4, 2)
        ]
        for u, v, w in edges:
            self.graph.add_edge(u, v, w)

    def test_naive_dijkstra(self):
        """Test naive Dijkstra implementation"""
        result = DijkstraImplementations.naive_dijkstra(self.graph, 0)
        expected = {0: 0, 1: 3, 2: 2, 3: 8, 4: 10}
        self.assertEqual(result, expected)

    def test_binary_heap_dijkstra(self):
        """Test binary heap Dijkstra implementation"""
        result = DijkstraImplementations.binary_heap_dijkstra(self.graph, 0)
        expected = {0: 0, 1: 3, 2: 2, 3: 8, 4: 10}
        self.assertEqual(result, expected)

    def test_fibonacci_heap_dijkstra(self):
        """Test Fibonacci heap Dijkstra implementation"""
        result = DijkstraImplementations.fibonacci_heap_dijkstra(self.graph, 0)
        expected = {0: 0, 1: 3, 2: 2, 3: 8, 4: 10}
        self.assertEqual(result, expected)

    def test_all_implementations_consistency(self):
        """Test that all implementations produce identical results"""
        start = 0
        naive_result = DijkstraImplementations.naive_dijkstra(self.graph, start)
        binary_result = DijkstraImplementations.binary_heap_dijkstra(self.graph, start)
        fib_result = DijkstraImplementations.fibonacci_heap_dijkstra(self.graph, start)
        
        self.assertEqual(naive_result, binary_result)
        self.assertEqual(binary_result, fib_result)

    def test_single_vertex_graph(self):
        """Test single vertex graph"""
        single_graph = Graph()
        single_graph.vertices.add(0)
        
        result = DijkstraImplementations.binary_heap_dijkstra(single_graph, 0)
        self.assertEqual(result[0], 0)

    def test_disconnected_graph(self):
        """Test disconnected graph"""
        disconnected_graph = Graph()
        disconnected_graph.add_edge(0, 1, 1)
        disconnected_graph.add_edge(2, 3, 1)
        
        result = DijkstraImplementations.binary_heap_dijkstra(disconnected_graph, 0)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)
        self.assertEqual(result[2], float('inf'))
        self.assertEqual(result[3], float('inf'))

    def test_weighted_graph_paths(self):
        """Test complex weighted graph paths"""
        complex_graph = Graph()
        edges = [
            (0, 1, 10), (0, 2, 3), (1, 2, 1), (1, 3, 2),
            (2, 1, 4), (2, 3, 8), (2, 4, 2), (3, 4, 7),
            (4, 3, 9)
        ]
        for u, v, w in edges:
            complex_graph.add_edge(u, v, w)
        
        result = DijkstraImplementations.binary_heap_dijkstra(complex_graph, 0)
        # Verify some key shortest paths
        self.assertEqual(result[0], 0)  # Start vertex
        self.assertEqual(result[2], 3)  # Direct path
        self.assertEqual(result[4], 5)  # Via vertex 2

class TestGraphClass(unittest.TestCase):
    """Test the Graph class functionality"""

    def setUp(self):
        self.graph = Graph()

    def test_add_edge(self):
        """Test adding edges to graph"""
        self.graph.add_edge(0, 1, 5)
        self.assertIn(0, self.graph.vertices)
        self.assertIn(1, self.graph.vertices)
        self.assertIn((1, 5), self.graph.get_neighbors(0))
        self.assertIn((0, 5), self.graph.get_neighbors(1))

    def test_get_vertices(self):
        """Test getting all vertices"""
        self.graph.add_edge(0, 1, 1)
        self.graph.add_edge(1, 2, 2)
        vertices = self.graph.get_vertices()
        self.assertEqual(set(vertices), {0, 1, 2})

    def test_get_neighbors(self):
        """Test getting neighbors of a vertex"""
        self.graph.add_edge(0, 1, 3)
        self.graph.add_edge(0, 2, 5)
        neighbors = self.graph.get_neighbors(0)
        self.assertEqual(set(neighbors), {(1, 3), (2, 5)})

    def test_empty_graph(self):
        """Test empty graph operations"""
        self.assertEqual(len(self.graph.get_vertices()), 0)
        self.assertEqual(len(self.graph.get_neighbors(0)), 0)

class TestFibonacciHeap(unittest.TestCase):
    """Test Fibonacci heap operations"""

    def setUp(self):
        self.fib_heap = FibonacciHeap()

    def test_insert_and_extract_min(self):
        """Test basic insert and extract min operations"""
        self.fib_heap.insert(5, 'A')
        self.fib_heap.insert(3, 'B')
        self.fib_heap.insert(7, 'C')
        
        self.assertFalse(self.fib_heap.is_empty())
        
        min_key, min_value = self.fib_heap.extract_min()
        self.assertEqual(min_key, 3)
        self.assertEqual(min_value, 'B')

    def test_decrease_key(self):
        """Test decrease key operation"""
        self.fib_heap.insert(5, 'A')
        self.fib_heap.insert(3, 'B')
        self.fib_heap.insert(7, 'C')
        
        # Decrease key of C from 7 to 1
        result = self.fib_heap.decrease_key('C', 1)
        self.assertTrue(result)
        
        min_key, min_value = self.fib_heap.extract_min()
        self.assertEqual(min_key, 1)
        self.assertEqual(min_value, 'C')

    def test_empty_heap(self):
        """Test empty heap behavior"""
        self.assertTrue(self.fib_heap.is_empty())
        result = self.fib_heap.extract_min()
        self.assertIsNone(result)

    def test_decrease_key_invalid(self):
        """Test invalid decrease key operations"""
        self.fib_heap.insert(5, 'A')
        
        # Try to increase key (should fail)
        result = self.fib_heap.decrease_key('A', 10)
        self.assertFalse(result)
        
        # Try to decrease key of non-existent node
        result = self.fib_heap.decrease_key('B', 1)
        self.assertFalse(result)

class TestGraphGenerator(unittest.TestCase):
    """Test graph generation utilities"""

    def test_random_graph_generation(self):
        """Test random graph generation"""
        graph = GraphGenerator.generate_random_graph(10, 15)
        self.assertGreaterEqual(len(graph.get_vertices()), 10)
        
        # Count edges
        edge_count = sum(len(graph.get_neighbors(v)) for v in graph.get_vertices()) // 2
        self.assertGreaterEqual(edge_count, 9)  # At least spanning tree

    def test_sparse_graph_generation(self):
        """Test sparse graph generation"""
        graph = GraphGenerator.generate_sparse_graph(10)
        self.assertGreaterEqual(len(graph.get_vertices()), 10)
        
        # Sparse graphs should have roughly 2V edges
        edge_count = sum(len(graph.get_neighbors(v)) for v in graph.get_vertices()) // 2
        self.assertLessEqual(edge_count, 30)  # Should be sparse

    def test_dense_graph_generation(self):
        """Test dense graph generation"""
        graph = GraphGenerator.generate_dense_graph(10)
        self.assertGreaterEqual(len(graph.get_vertices()), 10)
        
        # Dense graphs should have many edges
        edge_count = sum(len(graph.get_neighbors(v)) for v in graph.get_vertices()) // 2
        self.assertGreaterEqual(edge_count, 20)  # Should be dense

class TestPerformanceAnalyzer(unittest.TestCase):
    """Test performance analysis utilities"""

    def test_measure_execution_time(self):
        """Test execution time measurement"""
        def dummy_function():
            time.sleep(0.01)  # Sleep for 10ms
            return "result"
        
        exec_time, result = PerformanceAnalyzer.measure_execution_time(dummy_function)
        self.assertGreaterEqual(exec_time, 0.009)  # Should be at least 9ms
        self.assertEqual(result, "result")

    def test_analyze_small_graphs(self):
        """Test analysis on small graphs"""
        # Test with very small graphs to avoid long test times
        small_sizes = [5, 10]
        results = PerformanceAnalyzer.analyze_implementations(
            small_sizes, ['sparse']
        )
        
        self.assertEqual(results['sizes'], small_sizes)
        self.assertEqual(len(results['naive']['sparse']), 2)
        self.assertEqual(len(results['binary_heap']['sparse']), 2)
        self.assertEqual(len(results['fibonacci_heap']['sparse']), 2)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_main_functionality(self):
        """Test that main components work together"""
        # Create a test graph
        graph = GraphGenerator.generate_random_graph(20, 30)
        
        # Run all algorithms
        start = 0
        naive_result = DijkstraImplementations.naive_dijkstra(graph, start)
        binary_result = DijkstraImplementations.binary_heap_dijkstra(graph, start)
        fib_result = DijkstraImplementations.fibonacci_heap_dijkstra(graph, start)
        
        # All should produce the same result
        self.assertEqual(naive_result, binary_result)
        self.assertEqual(binary_result, fib_result)
        
        # Start vertex should have distance 0
        self.assertEqual(naive_result[start], 0)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plotting_functionality(self, mock_savefig, mock_show):
        """Test that plotting works without displaying"""
        # Create mock results
        results = {
            'sizes': [10, 20],
            'naive': {'sparse': [0.001, 0.004], 'dense': [0.002, 0.008]},
            'binary_heap': {'sparse': [0.0005, 0.001], 'dense': [0.001, 0.003]},
            'fibonacci_heap': {'sparse': [0.001, 0.002], 'dense': [0.002, 0.005]}
        }
        
        # This should not raise an exception
        PerformanceAnalyzer.plot_results(results)
        
        # Verify that savefig was called
        mock_savefig.assert_called_once()

def run_test_suite():
    """Run the complete test suite with detailed output"""
    print("=" * 60)
    print("  DIJKSTRA ALGORITHM TEST SUITE")
    print("  Author: Kabore Taryam William Rodrigue")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDijkstraImplementations,
        TestGraphClass,
        TestFibonacciHeap,
        TestGraphGenerator,
        TestPerformanceAnalyzer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        return False

if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)
