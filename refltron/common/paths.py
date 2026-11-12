"""Where things live on disk, measured from the project folder.

Code files can sit at any depth inside refltron/, so they use PROJECT_ROOT instead of
counting parent folders from their own location.
"""

from pathlib import Path

# refltron/common/paths.py -> parents[0] = common, [1] = refltron, [2] = the project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
