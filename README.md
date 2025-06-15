# Dijkstra's Algorithm Performance Analysis

## Project Overview
This project implements and compares three different versions of Dijkstra's shortest path algorithm:
1. **Naive Implementation** - O(V²) time complexity
2. **Binary Heap Implementation** - O((V + E) log V) time complexity  
3. **Fibonacci Heap Implementation** - O(E + V log V) time complexity

## Features
- Complete implementation of all three algorithmic approaches
- Performance analysis on both sparse and dense graphs
- Comprehensive test suite for correctness verification
- Visual performance comparison with matplotlib
- Memory usage analysis
- Real-world and synthetic graph testing

## Algorithm Complexity Analysis

### Time Complexities
| Implementation | Time Complexity   | Space Complexity |
|----------------|-------------------|------------------|
| Naive          | O(V²) | O(V)      |
| Binary Heap    | O((V + E) log V)  |    O(V)          |
| Fibonacci Heap | O(E + V log V)    |   O(V)           |

### When to Use Each Implementation
- **Naive**: Best for very small graphs (V < 100) or when simplicity is priority
- **Binary Heap**: Best general-purpose implementation, excellent performance/simplicity ratio
- **Fibonacci Heap**: Best for very dense graphs where E approaches V², theoretical optimal

## Project Structure
```
dijkstra-analysis/
├── main.py                    # Main implementation file
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── results/                   # Generated analysis results
│   └── dijkstra_performance_comparison.png
└── tests/                     # Test cases and validation
```

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- Memory profiler (optional)

## Installation & Setup

### Option 1: Local Setup
```bash
# Clone the repository
git clone [your-repo-url]
cd dijkstra-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python main.py
```

### Option 2: Replit Setup
1. Create new Python repl on Replit
2. Upload `main.py` to your repl
3. Install dependencies in Shell:
   ```bash
   pip install numpy matplotlib
   ```
4. Run the program:
   ```bash
   python main.py
   ```

## Usage

### Basic Usage
```python
from main import *

# Create a graph
graph = GraphGenerator.generate_random_graph(100, 300)

# Run different implementations
distances_naive = DijkstraImplementations.naive_dijkstra(graph, 0)
distances_binary = DijkstraImplementations.binary_heap_dijkstra(graph, 0)
distances_fib = DijkstraImplementations.fibonacci_heap_dijkstra(graph, 0)
```

### Performance Analysis
```python
# Analyze performance across different graph sizes
sizes = [50, 100, 200, 500, 1000]
results = PerformanceAnalyzer.analyze_implementations(sizes)
PerformanceAnalyzer.plot_results(results)
```

## Key Implementation Details

### Fibonacci Heap Features
- **Decrease Key**: O(1) amortized time
- **Extract Min**: O(log V) amortized time
- **Insert**: O(1) time
- **Merge**: O(1) time

### Binary Heap (Python heapq)
- **Extract Min**: O(log V) time
- **Insert**: O(log V) time
- Simple to implement and debug
- Good cache performance

### Graph Generation
- **Random Graphs**: Configurable vertex count and edge density
- **Sparse Graphs**: ~2V edges for testing linear-time performance
- **Dense Graphs**: ~V²/2 edges for testing quadratic scenarios

## Testing & Validation

The project includes comprehensive testing:

### Correctness Tests
- Verifies all implementations produce identical results
- Tests edge cases (single vertex, disconnected graphs)
- Validates against known shortest path solutions

### Performance Tests
- Measures execution time across different graph sizes
- Compares memory usage between implementations
- Tests both sparse and dense graph scenarios

## Expected Results

### Performance Characteristics
1. **Small Graphs (V < 100)**: All implementations perform similarly
2. **Medium Graphs (100 < V < 1000)**: Binary heap shows clear advantage over naive
3. **Large Dense Graphs (V > 1000, E >> V)**: Fibonacci heap demonstrates theoretical superiority
4. **Large Sparse Graphs**: Binary heap often performs best in practice

### Complexity Verification
The analysis will demonstrate:
- Naive implementation: Quadratic growth with vertex count
- Binary heap: Near-linear growth for sparse graphs
- Fibonacci heap: Optimal performance for dense graphs

## Output Files
- `dijkstra_performance_comparison.png`: Performance comparison charts
- Console output: Detailed timing and correctness verification

## Academic Context
This implementation demonstrates:
- Advanced data structure usage (Fibonacci heaps)
- Algorithm optimization techniques
- Empirical algorithm analysis
- Graph theory applications

## Contributing
Feel free to extend this project by:
- Adding more graph types (planar, bipartite, etc.)
- Implementing A* algorithm comparison
- Adding parallel processing capabilities
- Including real-world graph datasets

## License
MIT License - feel free to use for educational purposes.

## Author
Kabore Taryam William Rodrigue 

________________________________________
Project - Comparative Analysis of Dijkstra's Algorithm Implementations

