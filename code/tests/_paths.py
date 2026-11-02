"""Shared import paths for the research-audit test suite (stdlib unittest only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in [ROOT / 'code' / 'audit', ROOT / 'code' / 'revision_20261022', ROOT / 'code']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.dont_write_bytecode = True
