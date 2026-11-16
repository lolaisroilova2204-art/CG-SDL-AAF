#!/usr/bin/env bash
# Full benchmark for the manuscript (LPIPS + Blender). Requires: pip install -e .[lpips]; Blender on PATH or CGAAF_BLENDER set.
set -e
python -c "import lpips; lpips.LPIPS(net='alex')"
cgaaf benchmark benchmark_full --repeats 3
cgaaf paper --benchmark benchmark_full --out paper_tech
echo "results: paper_tech/paper_results_en.md, paper_tech/paper_tables.xlsx, paper_tech/figures/"
