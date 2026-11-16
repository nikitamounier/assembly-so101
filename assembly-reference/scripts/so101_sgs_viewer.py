"""Export or serve the read-only SGS state browser (see tools/sgs_viewer/README.md)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assembly_policy.so101.sgs_viewer import main

if __name__ == "__main__":
    main()
