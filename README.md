Dijkstra's Algorithm: Comparative Analysis
This project implements and compares three different versions of Dijkstra's shortest path algorithm to analyze their performance characteristics.
📋 Project Overview
This  project demonstrates the implementation and analysis of three Dijkstra's algorithm variations:
•	Naive Implementation - O(V²) time complexity
•	Binary Heap Implementation - O((V+E)logV) time complexity
•	Fibonacci Heap Implementation - O(VlogV + E) time complexity
🎯 Project Goals
•	Implement three variations of Dijkstra's algorithm
•	Compare performance across different graph sizes
•	Analyze time complexity differences
•	Visualize execution time comparisons
•	Document findings and conclusions

📁 Project Files
dijkstra-midterm-project/
├── main.py                         # Main implementation file
├── test_dijkstra.py                # Algorithm testing suite
├── dijkstra_performance_comparison.png  # Performance visualization
├── requirements.txt                # Python dependencies
├── setup.py                       # Project setup configuration
├── run_analysis.sh                # Analysis execution script
└── README.md                      # This documentation


🚀 How to Run
Requirements
•	Python 3.7 or higher
•	matplotlib (for visualizations)
•	numpy (for calculations)
Installation
# Install dependencies
pip install -r requirements.txt

# Or run setup
python setup.py install
Running the Project
1.	Run the main analysis:
2.	python main.py
3.	Run tests:
4.	python test_dijkstra.py
5.	Run complete analysis (Linux/Mac):
6.	./run_analysis.sh
📊 Results Summary
Test Configuration
•	Graph sizes tested: 100, 500, 1000, 2000 vertices
•	Multiple test runs for accuracy
•	Execution time measured in seconds
Key Findings
Algorithm	Time Complexity	Best for
Naive	O(V²)	Small graphs
Binary Heap	O((V+E)logV)	Medium graphs
Fibonacci Heap	O(VlogV + E)	Large, sparse graphs
Performance Charts
Performance comparison visualization is saved as dijkstra_performance_comparison.png.
🔍 Algorithm Details
Naive Implementation
•	Simple approach using arrays
•	Good for small graphs due to low overhead
•	Becomes inefficient as graph size increases
Binary Heap Implementation
•	Uses Python's heapq module
•	Significant improvement for medium-sized graphs
•	Good balance of performance and simplicity
Fibonacci Heap Implementation
•	Most complex but theoretically optimal
•	Best performance on large, sparse graphs
•	Higher constant factors may affect small graphs
🧪 Testing
The project includes comprehensive testing:
•	Correctness Testing: Verify all implementations produce correct shortest paths
•	Performance Testing: Compare execution times across different graph sizes
•	Edge Case Testing: Handle disconnected graphs and special cases
📈 Performance Analysis
The analysis includes:
•	Execution time measurements
•	Complexity growth visualization
•	Memory usage comparison
•	Algorithm efficiency analysis
🛠️ Technologies Used
•	Python 3.x - Core implementation
•	NumPy - Numerical computations
•	Matplotlib - Data visualization
•	heapq - Binary heap operations
•	time - Performance measurement
________________________________________
Project - Comparative Analysis of Dijkstra's Algorithm Implementations

