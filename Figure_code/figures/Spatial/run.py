#!/usr/bin/env python3
"""Render Spatial rankings, STC/SVC, resources, downsampling, and RNA comparisons."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_benchmark import main
if __name__ == '__main__':
    main('Spatial')
