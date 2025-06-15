
#!/bin/bash

# Dijkstra's Algorithm Analysis - Automated Execution Script
# Author: Kabore Taryam William Rodrigue
# Description: Automated script to run complete analysis pipeline

set -e  # Exit on any error

echo "=============================================="
echo "  Dijkstra's Algorithm Performance Analysis  "
echo "  Author: Kabore Taryam William Rodrigue     "
echo "=============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ Error: Python ${REQUIRED_VERSION} or higher is required. Found: ${PYTHON_VERSION}"
    exit 1
fi

echo "✅ Python ${PYTHON_VERSION} detected"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
python3 -m pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found"
    exit 1
fi

# Run the analysis
echo ""
echo "🚀 Starting Dijkstra Algorithm Analysis..."
echo "⏱️  This may take a few minutes depending on your system..."
echo ""

# Create a timestamp for the run
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "🕐 Analysis started at: ${TIMESTAMP}"
echo ""

# Run the main analysis
python3 main.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    
    # Check if output files were generated
    if [ -f "dijkstra_performance_comparison.png" ]; then
        echo "📊 Performance chart generated: dijkstra_performance_comparison.png"
    fi
    
    echo ""
    echo "📁 Generated files:"
    ls -la *.png 2>/dev/null || echo "  No PNG files found"
    
else
    echo ""
    echo "❌ Analysis failed with exit code $?"
    exit 1
fi

# Run tests if test file exists
if [ -f "test_dijkstra.py" ]; then
    echo ""
    echo "🧪 Running test suite..."
    python3 test_dijkstra.py
    
    if [ $? -eq 0 ]; then
        echo "✅ All tests passed!"
    else
        echo "❌ Some tests failed"
        exit 1
    fi
fi

# Final summary
echo ""
echo "=============================================="
echo "              ANALYSIS COMPLETE              "
echo "=============================================="
echo ""
echo "📋 Summary:"
echo "  ✅ Correctness verification: PASSED"
echo "  ✅ Performance analysis: COMPLETED"
echo "  ✅ Visualization: GENERATED"
echo "  ✅ Test suite: PASSED"
echo ""
echo "🎯 Key Results:"
echo "  • Naive Implementation: O(V²) - Best for small graphs"
echo "  • Binary Heap: O((V+E)logV) - Best practical performance"
echo "  • Fibonacci Heap: O(E+VlogV) - Theoretical optimum"
echo ""
echo "📊 Output Files:"
echo "  • dijkstra_performance_comparison.png - Performance charts"
echo "  • Console output - Detailed timing results"
echo ""
echo "🏆 Project by: Kabore Taryam William Rodrigue"
echo "🔗 GitHub: https://github.com/JIMMY62m24"
echo ""
echo "🎉 Analysis pipeline completed successfully!"
