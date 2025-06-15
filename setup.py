
#!/usr/bin/env python3
"""
Setup script for Dijkstra Algorithm Analysis Project
Author: Kabore Taryam William Rodrigue
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dijkstra-algorithm-analysis",
    version="1.0.0",
    author="Kabore Taryam William Rodrigue",
    author_email="jimmy62m24@example.com",
    description="Comprehensive implementation and performance analysis of Dijkstra's shortest path algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JIMMY62m24/dijkstra-algorithm-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/JIMMY62m24/dijkstra-algorithm-analysis/issues",
        "Documentation": "https://github.com/JIMMY62m24/dijkstra-algorithm-analysis#readme",
        "Source Code": "https://github.com/JIMMY62m24/dijkstra-algorithm-analysis",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dijkstra-analysis=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "dijkstra",
        "algorithm",
        "graph",
        "shortest-path",
        "performance-analysis",
        "data-structures",
        "fibonacci-heap",
        "binary-heap",
        "computer-science",
        "education",
    ],
)
