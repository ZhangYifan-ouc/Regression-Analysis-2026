"""
Wrapper script to run week07 experiment.
"""
from pathlib import Path
import sys

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run week07 main
from week07.main import main

if __name__ == "__main__":
    main()
