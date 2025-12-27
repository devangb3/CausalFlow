import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from experiments.skill_decomposition import main


if __name__ == "__main__":
    # Override with --experiment SealQA or --experiment MedBrowseComp if needed.
    main(default_experiment="BrowseComp")
