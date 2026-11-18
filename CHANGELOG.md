# Changelog

## 1.4.0 — 2026-11-16 (revision after the technical review of the manuscript)
### Changed
- Image channel: explicit **object mode** (`trim = True`, default; placement and size not assessed) and **composition mode** (`trim = False`); the benchmark covers both (1.3 benchmarked the composition mode only, which did not match the task default).
- Topology is penalised only in excess of the reference rate (tolerance max(2, 0.005 E)).
- Helper-object / helper-component rule no longer depends on the face count: flat (< 1 %) and ≥ 0.8 × the rest, or slab (< 3 %) and ≥ 1.5 × the rest; components are removed with a face mask so that welded vertices are preserved; small thin parts of the model are kept.
- Viewport detection for light interface themes; median corner background; trimming ignores a 2-px border.
- Feedback: ties of contour precision and recall count as missing contours; a penalised result (score < 86) without a specific message receives the region (image) or overall-shape (mesh) message.
- GUI label of the `trim` option now names the object / composition mode.
### Security
- Blender runs with `--disable-autoexec` and opens files with `use_scripts=False`.
### Fixed
- Chamfer "without alignment" baseline compared an unrotated student cloud with a PCA-rotated reference; both are now in their original orientations.
### Benchmark
- 12 Blender compositions (6 held-out, created after the defaults were fixed), student-like render settings, independent re-tessellation, viewport (Workbench) screenshots, 512 px check, greyscale and reframing families.
- Baselines added: FLIP, SNCD, Chamfer after ICP only and after PCA only, Hausdorff, D2 shape distribution.
- Statistics: paired bootstrap ΔAUC on reference-object resamples, leave-one-object-out ranges, development vs held-out AUC; Cohen's d dropped.
- Feedback validation (hit rates overall and among penalised cases, false alarms, occlusion localisation) and sensitivity analysis of all default constants (`cgaaf/sensitivity.py`).
- Reference run in `docs/benchmark_reference_run/` (Blender 5.3.0, LPIPS not evaluated); `bench_image.csv` and `bench_mesh.csv` keep the messages of the rules as specified in `feedback_initial`.
### Added
- `scripts/paper/make_figures.py`, `scripts/paper/manuscript_numbers.py`, `scripts/time_framework.py`, `docs/paper_figures/`, `tests/test_feedback_rules.py`.

## 1.3.0
- Technical-validation benchmark: six parametric Blender compositions, nuisance vs error families, AUC with reference-cluster bootstrap, render-level comparison of screenshots and re-renders; `cgaaf paper --benchmark` without a course.

## 1.2.0
- Interface removal from screenshots (+ manual crop), helper objects excluded, animation frames, optional size check, expert-grade scales, student-cluster bootstrap, version information in every result, `requirements-lock.txt`.
