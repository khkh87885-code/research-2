"""Use python -I run.py ... to isolate the inherited host PYTHONPATH."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(not result.wasSuccessful())
    from research_platform.cli import main
    main()
