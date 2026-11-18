# CG-SDL-AAF — Method specification (for the manuscript)

Algorithm version **CG-SDL-AAF 1.4** (package 1.4.0). Every symbol below maps one-to-one to the code (`cgaaf/`).

## 1. Self-directed learning cycle
Each attempt *a* of student *s* on assignment *t* records
**forethought** (goal text, planned minutes, self-predicted score *ŝ*), **performance** (submitted files → automated score *S* and feedback) and **self-reflection** (free text).
Calibration bias *b = ŝ − S* and absolute error *e = |b|* (absolute accuracy in Schraw's terms) are shown to the student: accurate (*e* ≤ 10), over-confident (*b* > 10) or under-confident (*b* < −10). The change of *e* between the first and last attempts is a candidate indicator of developing self-monitoring; its effect on learning has not been evaluated.

## 2. Submission handling and Blender re-rendering
| Input | Image channel | Geometry channel |
|---|---|---|
| `.blend`, `.fbx` | headless Blender render (`render.png`) | evaluated meshes exported (`export.obj`, modifiers applied, triangulated) |
| PNG/JPG | used directly (*screenshot*) | — |
| OBJ/STL/PLY/OFF/GLB | — | loaded with trimesh, vertices welded (10⁻⁶) |

**Security.** Blender is started with `--disable-autoexec` and files are opened with `use_scripts=False`, so Python code embedded in a student's file is never executed.

Canonical render (`mode = canonical`): bounding box of all assessed objects → centre **c**, radius *ρ*; perspective camera, 50 mm lens / 36 mm sensor, azimuth 45°, elevation 25°, distance *d = 1.05 ρ / sin(fov/2)*; Cycles on CPU, fixed seed 0, no denoising, opaque film, 100 % resolution, "Standard" view transform — whatever the student's file specifies (engine, view transform, camera, film transparency, resolution percentage and denoising are all overridden); square resolution (default 512 px in courses, 256 px in the benchmark; 32 samples). A sun lamp is added only if the scene has no light. The **reference .blend passes through exactly the same pipeline**, which removes camera/framing/render-setting variance between reference and submissions.

**Assessed objects** (`objects`): `auto` excludes helper objects — an object is a helper if it is *flat* (thickness < 1 % of its largest extent) and at least 0.8 × as large as the rest of the model, or a *thin slab* (< 3 %) at least 1.5 × as large; objects with a helper name (*Plane/Floor/Ground/Backdrop*) thinner than 5 % are treated alike. The rule does not depend on the number of faces (subdivided floors are recognised) and keeps small thin parts of the model (plates, signs). Alternatively `names:` patterns, a `collection:` or `all`. Only assessed objects are exported, used for camera framing and — unless `render_helpers` is set — rendered. For mesh files without object information, connected components are tested with the same rule and removed with a face mask, so that welded vertices are preserved.
**Animation** (`frames = K > 1`): the reference and each submission are rendered at K frames evenly spaced over [frame_start, frame_end] (normalised time); per-frame image metrics are averaged and the worst frame is reported. Geometry is exported at frame_start.

## 3. Perceptual image metrics (`image_metrics.py`)
0. **UI removal** (`ui_crop.py`, screenshots only). A recursive XY-cut splits the screenshot along thin (1–6 px) uniform lines that separate editor areas: very dark lines (luminance < 36) in dark themes, and lines whose grey differs by ≥ 25 levels from the rows ±4 px away on both sides in light themes. Among the leaf regions, the largest one that is not an interface panel (an unsaturated region whose three most frequent quantised colours cover ≥ 97 % of it) is taken as the 3D viewport, and a uniform header/footer band of 10–40 px is trimmed. Clean renders contain no such separators and are left untouched. A manual crop overrides the automatic one; the applied box is recorded in `result.json`. When several screenshots are submitted, the best-matching one is assessed. *The light-theme detector is unreliable (manuscript, Table XIII).*
1. **Normalisation.** Background colour **b** = median of four corner patches. **Object mode** (`trim = True`, default): both images are trimmed to the bounding box of pixels with ‖x − b‖₁ > 36 (+5 % margin, ignoring a 2-px border), so placement and size are not assessed. **Composition mode** (`trim = False`): frames are compared as they are (framing, roll and placement count). Then aspect-preserving fit into *N × N* (*N* = 256, Lanczos), padding with **b**.
2. **Foreground weight.** *F* = {‖R − b‖₁ > 30} ∪ {‖S − b‖₁ > 30}, dilated by *N*/32 px; *w* = 1 on *F*, 0.1 elsewhere.
3. **SSIM** on luminance *Y = 0.299R + 0.587G + 0.114B*: Gaussian window σ = 1.5 (11 × 11), *C₁ = (0.01·255)², C₂ = (0.03·255)²*; reported value is the *w*-weighted mean of the SSIM map.
4. **MS-SSIM:** 4 dyadic scales (256 → 32), weights (0.0448, 0.2856, 0.3001, 0.2363) renormalised.
5. **LPIPS** (Zhang et al., 2018), AlexNet backbone; *m_lpips = clip(1 − d/0.6, 0, 1)*. Optional; if unavailable the weight is redistributed (it was unavailable in the reference run).
6. **Colour fidelity:** *m_color = ½ BC + ½ exp(−ΔE₀₀/12)*; BC = Bhattacharyya coefficient of 170-bin HSV histograms (18 hue × 3 sat × 3 val + 8 grey), ΔE₀₀ = *w*-weighted mean CIEDE2000 on 64 × 64 block means.
7. **Edge F1:** Sobel magnitude, threshold *T = max(60, P₉₀)*, precision/recall with 2-px tolerance.
8. Formative localisation: 3 × 3 grid cell with the lowest mean SSIM.

## 4. Geometric mesh metrics (`mesh_metrics.py`)
**Algorithm 1 — mesh comparison**
```
input : student mesh M; reference cloud Q and topology counts (prepared once with steps 1-5)
1  topology(M): boundary edges B, non-manifold edges N, degenerate faces D, counted in excess of the reference
   rate minus a tolerance max(2, 0.005 E)
2  P <- area-weighted surface sample of M, n = 4096, fixed seed
3  P <- (P - mean P) / RMS radius
4  if align = pca_icp:  P <- PCA canonical pose (det = +1, signs by third moment);
                        keep the proper axis flip with minimal Chamfer(P, Q) on 512-point subsamples
5  if align != none:    (R, t) <- ICP(P, Q), <= 30 iterations, k-d tree neighbours, Kabsch/SVD update
6  d_PQ, d_QP <- nearest-neighbour distances
7  CD = 1/2 (mean d_PQ + mean d_QP);  HD = max(max d_PQ, max d_QP);  H95 = max(P95(d_PQ), P95(d_QP))
8  precision = P(d_PQ < tau), recall = P(d_QP < tau), F = 2PR/(P+R),  tau = 0.1
```
Optional **absolute size** check (`check_size`): ratio *r* of RMS radii before normalisation, *m_size = exp(−|ln r|/0.10)* (weight 0.20).
Mapping to [0, 1]: *m_cd = exp(−max(0, CD − 0.03)/0.12)*, *m_h95 = exp(−max(0, H95 − 0.06)/0.30)*, *m_F = F*, *m_topo = clip(1 − (B + 2N + D)/(0.05 E), 0, 1)*. The floors 0.03 and 0.06 equal the resampling noise of an identical surface at *n* = 4096 (unit RMS radius).

## 5. Scoring model (`scoring.py`) and feedback (`feedback.py`)
Default: *P = Σ wₖ mₖ / Σ wₖ* over available metrics (image: SSIM 0.15, MS-SSIM 0.30, LPIPS 0.25, colour 0.15, edge 0.15; mesh: CD 0.40, H95 0.20, F 0.30, topology 0.10; image + mesh: *P = α P_img + (1 − α) P_mesh*, α = 0.5).
*Score = clip(100 (P − P₀)/(P₁ − P₀), 0, 100)*, P₀ = 0.40, P₁ = 0.98. Grade (5-point national scale): ≥ 86 → 5, 71–85 → 4, 55–70 → 3, < 55 → 2. The defaults were fixed a priori in version 1.3 and have not been re-tuned.
**Calibration:** *Score = b₀ + Σ bₖ mₖ*, *bₖ ≥ 0*, fitted by NNLS (Lawson–Hanson) on centred data with a free intercept; generalisation estimated by LOOCV.
**Feedback rules** (Uzbek/English): MS-SSIM < 0.75 → composition/shape; LPIPS > 0.35 → appearance; *m_color* < 0.7 → colour, materials or lighting; edge F1 < 0.6 → missing contours (recall ≤ precision) or extra contours; worst-cell SSIM < 0.8 → region; *m_cd* < 0.7 → overall shape; *m_h95* < 0.7 → outliers; precision < recall − 0.1 → extra parts (reverse → missing parts); excess *B*, *N*, *D* → holes, non-manifold edges, degenerate faces. **Version 1.4 corrections** found by the feedback validation: ties of contour precision and recall (no contour detected at all) count as missing contours, and a result graded below 5 (score < 86) without any specific message receives the region message (images) or the overall-shape message (meshes). *The extra-parts message remains unreliable (hit rate 4 % in the reference run).*

## 6. Validation statistics with expert ratings (`stats.py`, `reports.py`)
**Independence.** By default only the latest expert-rated attempt of each student on each assignment enters the validation, and all bootstrap intervals resample *students*.
Pearson *r* (Fisher *z* CI), Spearman ρ, MAE, RMSE, ICC(2,1) and ICC(3,1) (Shrout & Fleiss), percentile-bootstrap CIs (2000 resamples), Bland–Altman bias and limits of agreement, quadratic-weighted κ and exact agreement on the 5-point grade, inter-rater reliability, per-metric Spearman ρ with the expert score, LOOCV ablation. Learning gains: paired *t* and Wilcoxon tests, Cohen's *d_z*. *These routines are provided for the classroom validation that is still to come; the manuscript uses no student data.*

## 7. Technical validation: controlled benchmark (`benchmark.py`, `bench_render.py`, `discrimination.py`, `sensitivity.py`)
Six research questions without student data: **RQ1** monotonicity, **RQ2** invariance, **RQ3** discrimination vs baselines on development and held-out objects, **RQ4** evidence pipeline (re-render vs render and viewport screenshots), **RQ5** feedback validity, **RQ6** robustness, sensitivity, determinism and runtime.

*References.* Images: two procedural full-frame scenes + canonical renders of 11 Blender compositions. Meshes: torus, capsule, cylinder, ellipsoid + the main objects of 10 compositions. Render level: 12 compositions — development (existed when the defaults were fixed): torus + cube, Suzanne on a plinth, still life with a procedural texture, chair, mug, desk lamp; held-out (created afterwards): table + vase, bottle + glass, snowman, rocket, robot, kettle.

*Image channel* (both modes). Nuisance (severity ⅓, ⅔, 1): JPEG 95–85, Gaussian noise σ ≤ 3, 1–2 px shift, resampling to 60 %; object mode also reframing (70 %, shift ≤ 10 %; uniform-background references only). Errors (0.1–1): composition mode only translation ≤ 25 % and down-scaling ≤ 40 %; both modes rotation ≤ 25°, hue ≤ 180°, brightness −60 %, blur σ ≤ 6 px, noise σ ≤ 40, JPEG → 2, occlusion ≤ 35 %, greyscale ≤ 100 %.
*Mesh channel.* Nuisance: rigid rotation ≤ 180°, uniform scaling ×1–4 with translation, 1–2 loop subdivisions, micro-noise ≤ 0.2 % of the diagonal, independent re-tessellation (half / double segment counts). Errors: anisotropic scaling −60 %, vertex noise ≤ 3 %, hole ≤ 25 % of faces, part removal ≤ 40 %, extra box ≤ 35 % of the diagonal.
*Render level.* Reference: 32 samples, 256 px. Re-render nuisance: 8/16/64 samples at 256/384/192 px, three files saved with student-like render settings (EEVEE + AgX at 1920 × 1080; own camera + Filmic + 128 samples; transparent film at 50 % + denoising), two re-tessellated scenes. Screenshot pipelines: six jittered viewpoints (±8–22° azimuth, ±3–6° elevation, ±5–12 % distance) with the floor visible, Cycles (render screenshot) and Workbench solid shading (viewport screenshot). Errors: hue 0.04/0.10/0.25, vertical scale 0.9/0.75/0.5, light ×0.6/×0.35/×0.15, missing object, extra object. The re-render pipeline is repeated at 512 px.

*Discrimination.* AUC = P(s_nuisance > s_error) + ½ P(tie) for errors with severity ≥ 0.2 (Mann–Whitney); 95 % percentile CIs from 1000 bootstrap resamples of reference objects; the paired difference ΔAUC from the framework score on the same resamples decides whether a baseline differs (DeLong is not used because cases are clustered); leave-one-object-out ranges; AUC on development and held-out objects separately; detectability curves and the smallest severity with AUC ≥ 0.9. Cohen's *d* is no longer reported (nuisance scores of exactly 100 make it uninterpretable).
*Feedback validation.* Expected messages fixed per error family before the run; hit rate over all error cases and over cases graded below 5; false-alarm rate on nuisance cases; localisation of occlusions by the worst grid cell.
*Sensitivity.* Scores recomputed from the stored raw metrics with each weight ×0.5/×1.5, 200 Dirichlet weight vectors centred on the defaults, P₀ ∈ {0.30, 0.50}, P₁ ∈ {0.95, 1.00}, colour constant 6/24, floors and scales of the mesh mappings ×0.5/×1.5, τ ∈ {0.05, 0.15}, topology tolerance ×0.5/×2: range of the AUC and share of unchanged grades.
*Robustness.* (i) references and viewport screenshots embedded in three Blender-like layouts × 2 seeds, dark and light themes, scored with and without automatic cropping, crop IoU, false crops on 61 clean images; (ii) meshes combined with thin floors (3 ×, 4 ×, 6 × extent), a subdivided floor (768 faces) and a slab floor (2 %) against a reference with a thin 4 × floor, under the v1.3 rule, the v1.4 rule and no removal; control: a thin model plate (0.4 × extent, 0.3 %) omitted by the "student" must not be treated as a helper. Determinism: two re-renders compared pixel-wise and their OBJ exports byte-wise.

## 8. Baselines
Image: PSNR, unweighted SSIM, foreground-weighted SSIM, MS-SSIM, FLIP, SNCD (LZMA on 8-bit luminance), mean CIEDE2000, histogram BC, edge F1, and the unclipped composite *P*. Mesh: Chamfer without alignment (original orientations), after ICP only, after PCA only and after PCA + ICP, Hausdorff, H95, F-score and the D2 shape distribution (L1 distance of 64-bin histograms of 20,000 point-pair distances). With expert ratings, each baseline can also be mapped to the expert scale by a LOOCV linear model and compared by r, MAE, RMSE, ICC(2,1) and a paired-bootstrap CI of ΔMAE (`reports.py`).
Archived submissions graded before the framework existed are imported (`archive.py`); grades on other scales are rescaled to 0–100, national 2–5 grades are mapped to the centres of the score bands, and every attempt is flagged `retrospective`.

## 9. Known limitations (reference run of version 1.4)
* The benchmark is synthetic (author-made objects and single, idealised perturbations); it establishes the technical behaviour of the instrument, not agreement with teachers or effects on learning.
* Simple baselines matched or beat the composite: PSNR in the image channel and the Hausdorff distance in the geometry channel had slightly but significantly higher AUCs; the clipped score trades the detection of very small errors for tolerance.
* Screenshots are not suitable for grading: render and viewport screenshots of correct scenes were indistinguishable from erroneous ones (AUC 0.55 and 0.51); assess scene files.
* Light-theme interface removal is unreliable; crops are recorded and can be replaced manually.
* Greyscale renders of colour tasks usually keep the top grade under the default weights; raise the colour weight where colour matters.
* The extra-parts message is unreliable; the two v1.4 feedback corrections were evaluated on the same data that revealed the defects.
* LPIPS was not evaluated in the reference run (weights not downloadable in the build environment).
* Animation assessment compares appearance at K sampled frames; timing between samples, physics or rig quality are not assessed. `.max` files require export to FBX/OBJ.

## 10. Suggested reporting checklist
* Number of students, assignments, attempts; proportion of `.blend` re-renders vs screenshots; image mode per assignment.
* Expert rating protocol: blind to automated scores, rubric, two raters, inter-rater ICC.
* Agreement table (Section 6) per assignment and pooled; Bland–Altman and scatter figures.
* Ablation table; per-metric correlations; default vs calibrated model (LOOCV).
* SRL indicators: plan/reflection rates, *e* and *b* first → last attempt, learning gain.
* Software availability: repository URL, version 1.4.0, DOI (Zenodo), licence MIT.
