#!/usr/bin/env bash
set -euo pipefail
KBS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KBS_PYTHON="${KBS_PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
case "${1:-help}" in
  check|test|verify|prepare|analyze-cpu|replay-costs|figures|build)
    exec "$KBS_PYTHON" -B "$KBS_ROOT/code/project/run.py" "$@" ;;
  all)
    bash "$0" check
    bash "$0" replay-costs
    bash "$0" figures
    bash "$0" build ;;
  status)
    cat "$KBS_ROOT/docs/工作计划.md" ;;
  help|-h|--help)
    printf '%s\n' 'Usage: bash code/scripts/kbs_project.sh {check|test|verify|prepare|analyze-cpu|replay-costs|figures|build|all|status}' \
      'check: verify saved data and tables, then run correctness tests.' \
      'all: check, replay costs, export figures, and build the paper; no model requests.' \
      'Raw records are unpacked and verified under work/evidence/ only when needed.' ;;
  *) printf '%s\n' "Unknown command: $1" >&2; exit 2 ;;
esac
