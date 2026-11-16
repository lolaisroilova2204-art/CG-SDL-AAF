# CG-SDL-AAF — Method specification (for the manuscript)

Algorithm version **CG-SDL-AAF 1.3**. Every symbol below maps one-to-one to the code (`cgaaf/`).

## 1. Self-directed learning cycle
Each attempt *a* of student *s* on assignment *t* records
**forethought** (goal text, planned minutes, self-predicted score *ŝ*), **performance** (submitted files → automated score *S* and feedback) and **self-reflection** (free text).
Metacognitive calibration error: *|ŝ − S|*, compared between the first and last attempts (paired test).

## 2. Submission handling and Blender re-rendering
| Input | Image channel | Geometry channel |
|---|---|---|
| `.blend`, `.fbx` | headless Blender render (`render.png`) | evaluated meshes exported (`export.obj`, modifiers applied, triangulated) |
| PNG/JPG | used directly (*screenshot*) | — |
| OBJ/STL/PLY/OFF/GLB | — | loaded with trimesh, vertices welded (10⁻⁶) |

Canonical render (`mode = canonical`): bounding box of all renderable objects → centre **c**, radius *ρ*; perspective camera, 50 mm lens / 36 mm sensor, azimuth 45°, elevation 25°, distance *d = 1.05 ρ / sin(fov/2)*; Cycles on CPU, fixed seed 0, no denoising, “Standard” view transform, square resolution (default 512 px, 32 samples). A sun lamp is added only if the scene has no light. The **reference .blend passes through exactly the same pipeline**, which removes camera/framing variance between reference and submissions.

**Assessed objects** (`objects`): `auto` excludes plane-like helper objects (smallest dimension < 1 % of the largest with ≤ 64 polygons, or a helper name such as *Plane/Floor/Ground/Backdrop* with thickness < 5 %); alternatively `names:` patterns, a `collection:` or `all`. Only assessed objects are exported, used for camera framing and — unless `render_helpers` is set — rendered. For mesh files without object information, plane-like connected components (OBB thickness < 1 % of extent, ≤ 128 faces) are removed in the same spirit.
**Animation** (`frames = K > 1`): the reference and each submission are rendered at K frames evenly spaced over [frame_start, frame_end] (normalised time, so different animation lengths are comparable); per-frame image metrics are averaged and the worst frame is reported. Geometry is exported at frame_start.

## 3. Perceptual image metrics (`image_metrics.py`)
0. **UI removal** (`ui_crop.py`, screenshots only). A recursive XY-cut splits the screenshot along thin (1–6 px), uniform, dark (luminance < 36) lines that are darker than their ±8 px neighbourhood — the separators between editor areas of the Blender/3ds Max interface. The largest textured leaf region (area × [std > 6] × saturation factor) is taken as the 3D viewport, and a uniform unsaturated header/footer band of 10–40 px is trimmed. Clean renders contain no such separators and are left untouched. A manual crop stored for the submission overrides the automatic one; the applied box is recorded in `result.json`. When several screenshots are submitted, each is scored and the best-matching one is assessed.
1. **Normalisation.** Background colour **b** = mean of 4 × (4×4) corner patches. Optional trim to the bounding box of pixels with ‖x − b‖₁ > 36 (+5 % margin), aspect-preserving fit into *N × N* (*N* = 256, Lanczos), padding with **b**.
2. **Foreground weight.** *F* = {‖R − b‖₁ > 30} ∪ {‖S − b‖₁ > 30}, dilated by *N*/32 px; *w* = 1 on *F*, 0.1 elsewhere.
3. **SSIM** on luminance *Y = 0.299R + 0.587G + 0.114B*: Gaussian window σ = 1.5 (11 × 11), *C₁ = (0.01·255)², C₂ = (0.03·255)²*; reported value is the *w*-weighted mean of the SSIM map.
4. **MS-SSIM:** 4 dyadic scales (256 → 32), weights (0.0448, 0.2856, 0.3001, 0.2363) renormalised; MS = l₄^{β₄} ∏ csⱼ^{βⱼ}.
5. **LPIPS** (Zhang et al., 2018), AlexNet backbone, inputs scaled to [−1, 1]; *m_lpips = clip(1 − d/0.6, 0, 1)*. Optional; if unavailable the weight is redistributed.
6. **Colour fidelity:** *m_color = ½ BC + ½ exp(−ΔE₀₀/12)*; BC = Bhattacharyya coefficient of 170-bin HSV histograms (18 hue × 3 sat × 3 val + 8 grey), ΔE₀₀ = *w*-weighted mean CIEDE2000 on 64 × 64 block means.
7. **Edge F1:** Sobel magnitude, threshold *T = max(60, P₉₀)*, precision/recall with 2-px tolerance (5 × 5 dilation).
8. Formative localisation: 3 × 3 grid cell with the lowest mean SSIM.

## 4. Geometric mesh metrics (`mesh_metrics.py`)
**Algorithm 1 — mesh comparison**
```
input : student mesh M, reference cloud Q (prepared once with the same steps 2–4)
1  topology(M): boundary edges B (1 face), non-manifold edges N (>2 faces), degenerate faces D, components
2  P ← area-weighted surface sample of M, n = 4096, seed = 20260101
3  P ← (P − mean P) / RMS radius
4  if align = pca_icp:  P ← PCA canonical pose (eigenvectors of covariance, det = +1, signs by third moment)
                        P ← argmin over 4 proper axis flips of Chamfer(P, Q) on 512-point subsamples
5  if align ≠ none:     (R, t) ← ICP(P, Q), ≤ 30 iterations, nearest neighbours by k-d tree, Kabsch/SVD update
6  d_PQ, d_QP ← nearest-neighbour distances
7  CD = ½(mean d_PQ + mean d_QP);  HD = max(max d_PQ, max d_QP);  H95 = max(P₉₅(d_PQ), P₉₅(d_QP))
8  precision = P(d_PQ < τ), recall = P(d_QP < τ), F = 2PR/(P+R),  τ = 0.1
```
Optional **absolute size** check (`check_size`): ratio ρ = RMS radius(student)/RMS radius(reference) before normalisation, *m_size = exp(−|ln ρ|/0.10)* (weight 0.20).
Mapping to [0, 1]: *m_cd = exp(−max(0, CD − 0.03)/0.12)*, *m_h95 = exp(−max(0, H95 − 0.06)/0.30)*, *m_F = F*, *m_topo = clip(1 − (B + 2N + D)/(0.05 E), 0, 1)*. The floors 0.03 and 0.06 equal the resampling noise of an identical surface at *n* = 4096 (unit RMS radius).

## 5. Scoring model (`scoring.py`)
Default: *P = Σ wₖ mₖ / Σ wₖ* over available metrics (image: SSIM 0.15, MS-SSIM 0.30, LPIPS 0.25, colour 0.15, edge 0.15; mesh: CD 0.40, H95 0.20, F 0.30, topology 0.10; image + mesh: *P = α P_img + (1 − α) P_mesh*, α = 0.5).
*Score = clip(100 (P − P₀)/(P₁ − P₀), 0, 100)*, P₀ = 0.40, P₁ = 0.98. Grade (5-point national scale): ≥ 86 → 5, 71–85 → 4, 55–70 → 3, < 55 → 2.
**Calibration:** *Score = b₀ + Σ bₖ mₖ*, *bₖ ≥ 0*, fitted by NNLS (Lawson–Hanson) on centred data with a free intercept; generalisation estimated by leave-one-out cross-validation (LOOCV).

## 6. Validation statistics (`stats.py`, `reports.py`)
**Independence.** By default only the latest expert-rated attempt of each student on each assignment enters the validation, and all bootstrap intervals (ICC(2,1), MAE, r) resample *students* (cluster bootstrap), so that repeated observations of one student are not treated as independent in pooled analyses.
Pearson *r* (95 % CI, Fisher *z*), Spearman ρ, MAE, RMSE, ICC(2,1) absolute agreement and ICC(3,1) consistency (two-way ANOVA, Shrout & Fleiss), 95 % percentile-bootstrap CI for ICC(2,1) and MAE (2000 resamples, seed 1), Bland–Altman bias and limits of agreement (± 1.96 SD), quadratic-weighted Cohen's κ and exact agreement on the 5-point grade, inter-rater reliability between two experts, Spearman ρ of every raw metric with the expert score, and LOOCV ablation (each metric alone, all metrics, all-minus-one).
Learning gains: paired *t*-test and Wilcoxon signed-rank test (first vs best attempt), Cohen's *d_z*.

## 7. Technical validation: controlled benchmark (`benchmark.py`, `bench_render.py`, `discrimination.py`)
The benchmark answers five questions without student data.
**RQ1 (monotonicity)** — does the score fall as an error grows? **RQ2 (invariance)** — is it unaffected by nuisance variation a teacher would not penalise? **RQ3 (discrimination vs baselines)** — does it separate nuisance from error better than single metrics? **RQ4 (re-rendering)** — does standardised Blender re-rendering remove the viewpoint noise of screenshots? **RQ5 (robustness, reproducibility, cost)** — UI in screenshots, auxiliary objects, determinism, runtime.

*References.* Images: two procedural scenes + canonical renders of six Blender compositions (torus + cube; Suzanne on a plinth; a still life with a procedural noise texture; a chair; a mug; a desk lamp; metallic, glossy and rough materials). Meshes: torus, capsule, cylinder, ellipsoid + the Blender assets Suzanne (subdivided), chair, mug, desk lamp. Any user files can be added (`--ref-images`, `--ref-meshes`).

*Image channel.* Nuisance families (severity ⅓, ⅔, 1): JPEG quality 95–85, Gaussian noise σ ≤ 3, 1–2 px shift, down-/up-resampling to 60 %. Error families (severity .1–1): translation ≤ 25 % width, rotation ≤ 25°, down-scaling ≤ 40 %, hue ≤ 180°, brightness −60 %, blur σ ≤ 6 px, noise σ ≤ 40, JPEG quality → 2, occlusion ≤ 35 % of the shorter side.
*Mesh channel.* Nuisance: random rigid rotation, uniform scaling ×1–4 with translation, 1–2 loop subdivisions, micro-noise ≤ 0.2 % of the bounding-box diagonal. Errors: anisotropic scaling −60 %, vertex noise ≤ 3 %, a hole ≤ 25 % of faces, part removal ≤ 40 %, an extra box ≤ 35 % of the diagonal.
*Render level (Blender).* For each composition: reference (32 samples, 256 px); nuisance — re-renders with 8/16/64 samples at 256/384/192 px, and six viewpoint-jittered renders (±8–22° azimuth, ±3–6° elevation, ±5–12 % distance) simulating screenshots; errors — main-object hue shift 0.04/0.10/0.25, z-scale 0.9/0.75/0.5, light energy ×0.6/×0.35/×0.15, missing secondary object, extra object, each rendered canonically (re-render pipeline) and from a jittered viewpoint (screenshot pipeline). Determinism: two re-renders compared pixel-wise and their OBJ exports byte-wise.

*Discrimination measure.* For every method (framework score, the unclipped composite P, and each single metric oriented so that larger = more similar) AUC = P(s_nuisance > s_error) with mid-ranks for ties (equivalent to the Mann–Whitney U statistic), Cohen's d, and a 95 % percentile bootstrap CI that resamples reference objects (1000 resamples). Detectability curves give AUC against nuisance at each error severity, and the smallest severity with AUC ≥ 0.9.
*Robustness.* (i) references embedded in three Blender-like interface layouts × 3 seeds, scored with/without automatic cropping; (ii) references combined with an auxiliary floor (0, 3, 4, 6 × object extent; reference 4 ×), scored with/without helper-component removal.

## 8. Baselines and retrospective design
Single-metric baselines — PSNR, unweighted SSIM, MS-SSIM, LPIPS, ΔE₀₀ (image); Chamfer without pose alignment, Chamfer after PCA + ICP, Hausdorff-95 (mesh) — are each mapped to the expert scale by a linear model evaluated with LOOCV. They are compared with the framework (default weights, and NNLS-calibrated with LOOCV) by r, MAE, RMSE, ICC(2,1) and a paired-bootstrap 95 % CI of ΔMAE (2000 resamples).
Archived submissions and expert grades assigned before the framework existed are imported (`archive.py`); grades on other scales are rescaled linearly to 0–100 (*100·(g − g_min)/(g_max − g_min)*), or — for national 2–5 grades — mapped to the centre of the corresponding score band (5 → 93, 4 → 78, 3 → 62.5, 2 → 27.5) and every attempt is flagged `retrospective`. Because the grades were given without knowledge of any automated score, the validation is blind by construction.

*Robustness tests* — (i) the reference renders are embedded in three Blender-like interface layouts (3 seeds each) and scored with and without automatic cropping; (ii) each reference mesh is combined with an auxiliary floor (0, 3, 4, 6 × the object extent; reference 4 ×) and scored with and without helper-component removal.

## 9. Known limitations
* The benchmark uses synthetic, controlled perturbations and parametric scenes; it establishes the technical behaviour of the metrics (RQ1–RQ5), not agreement with human grading, which requires a classroom study with expert ratings (supported by the software, §8).
* The clipped 0–100 score deliberately tolerates deviations below the resampling noise floor; this costs some discrimination of very small errors relative to its own unclipped components.
* The UI detector was developed on synthetic Blender-like layouts (default dark theme); light themes, other software or heavily overlapped windows may defeat it — every crop is recorded and can be inspected or replaced manually (GUI: *Batch assessment → Manual crop*).
* Thresholds and mappings (P₀, P₁, metric floors, σ values, helper-object rules) are fixed a priori; the default score is not expert-validated and should be reported only after NNLS calibration.
* The default image score is weakly sensitive to blur and partial occlusion, and colour fidelity contributes 15 %; a greyscale copy of a colour render still scores high under default weights.
* Animation assessment compares appearance at K sampled frames; timing nuances between samples, physics or rig quality are not assessed.
* `.max` (3ds Max) files require export to FBX/OBJ; LPIPS requires PyTorch and a one-time download of AlexNet weights.

## 10. Suggested reporting checklist
* Number of students, assignments, attempts; proportion of `.blend` re-renders vs screenshots.
* Expert rating protocol: blind to automated scores, rubric, two raters, inter-rater ICC.
* Agreement table (Section 6) per assignment and pooled; Bland–Altman and scatter figures.
* Ablation table; per-metric correlations; default vs calibrated model (LOOCV).
* SRL indicators: plan/reflection rates, self-assessment error first → last attempt, learning gain.
* Software availability: repository URL, version 1.0, DOI (Zenodo), licence MIT.
