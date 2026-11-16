# Controlled-perturbation benchmark — CG-SDL-AAF 1.3

References: {'images': ['scene_A', 'scene_B', 'render_suzanne', 'render_still_life', 'render_chair', 'render_mug', 'render_desk_lamp'], 'meshes': ['torus', 'capsule', 'cylinder', 'ellipsoid', 'asset_suzanne', 'asset_chair', 'asset_mug', 'asset_desk_lamp'], 'error_levels': [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], 'nuisance_levels': [0.3333333333333333, 0.6666666666666666, 1.0], 'repeats': 3, 'lpips': False, 'render_compositions': ['chair', 'desk_lamp', 'mug', 'still_life', 'suzanne', 'torus_cube']}

## E1. Image channel — framework vs baselines (AUC, nuisance vs errors with severity ≥ 0.2)

| method                            |   AUC | AUC 95% CI     |   Cohen d |   n nuisance |   n error |
|:----------------------------------|------:|:---------------|----------:|-------------:|----------:|
| Edge F1                           | 0.934 | [0.907, 0.969] |     1.15  |          126 |       455 |
| PSNR                              | 0.919 | [0.902, 0.937] |     1.723 |          126 |       455 |
| Framework composite P (unclipped) | 0.916 | [0.896, 0.941] |     1.294 |          126 |       455 |
| MS-SSIM                           | 0.914 | [0.905, 0.929] |     1.039 |          126 |       455 |
| Framework score                   | 0.914 | [0.887, 0.938] |     1.272 |          126 |       455 |
| ΔE00                              | 0.891 | [0.847, 0.929] |     0.848 |          126 |       455 |
| SSIM (fg-weighted)                | 0.812 | [0.793, 0.839] |     0.831 |          126 |       455 |
| SSIM (unweighted)                 | 0.769 | [0.723, 0.828] |     0.673 |          126 |       455 |
| Colour histogram                  | 0.684 | [0.663, 0.703] |     0.689 |          126 |       455 |

## E2. Image channel — smallest severity detected with AUC ≥ 0.9

| method                            |   min severity with AUC≥0.9 |
|:----------------------------------|----------------------------:|
| Colour histogram                  |                       nan   |
| Edge F1                           |                         0.4 |
| Framework composite P (unclipped) |                         0.4 |
| Framework score                   |                         0.6 |
| MS-SSIM                           |                         0.4 |
| PSNR                              |                         0.4 |
| SSIM (fg-weighted)                |                         1   |
| SSIM (unweighted)                 |                       nan   |
| ΔE00                              |                         0.6 |

## E3. Mesh channel — framework vs baselines (AUC)

| method                            |   AUC | AUC 95% CI     |   Cohen d |   n nuisance |   n error |
|:----------------------------------|------:|:---------------|----------:|-------------:|----------:|
| Hausdorff                         | 0.994 | [0.992, 0.997] |     2.188 |          576 |       600 |
| Hausdorff-95                      | 0.993 | [0.990, 0.997] |     1.816 |          576 |       600 |
| Chamfer, PCA+ICP                  | 0.991 | [0.986, 0.997] |     1.993 |          576 |       600 |
| Framework composite P (unclipped) | 0.982 | [0.971, 0.992] |     1.885 |          576 |       600 |
| Framework score                   | 0.971 | [0.958, 0.983] |     1.837 |          576 |       600 |
| F-score@0.1                       | 0.967 | [0.955, 0.981] |     1.362 |          576 |       600 |
| Chamfer, no alignment             | 0.541 | [0.519, 0.621] |     0.116 |          576 |       600 |

## E4. Mesh channel — smallest severity detected

| method                            |   min severity with AUC≥0.9 |
|:----------------------------------|----------------------------:|
| Chamfer, PCA+ICP                  |                         0.1 |
| Chamfer, no alignment             |                       nan   |
| F-score@0.1                       |                         0.4 |
| Framework composite P (unclipped) |                         0.2 |
| Framework score                   |                         0.4 |
| Hausdorff                         |                         0.1 |
| Hausdorff-95                      |                         0.1 |

## E5. Render-level errors — re-render vs screenshot pipeline (AUC)

| pipeline   | method                            |   AUC | AUC 95% CI     |   Cohen d |   n nuisance |   n error |
|:-----------|:----------------------------------|------:|:---------------|----------:|-------------:|----------:|
| re-render  | ΔE00                              | 1     | [1.000, 1.000] |     1.455 |           18 |        66 |
| re-render  | Framework score                   | 0.985 | [0.958, 1.000] |     1.568 |           18 |        66 |
| re-render  | Framework composite P (unclipped) | 0.985 | [0.953, 1.000] |     1.574 |           18 |        66 |
| re-render  | PSNR                              | 0.982 | [0.957, 1.000] |     2.529 |           18 |        66 |
| re-render  | Colour histogram                  | 0.964 | [0.949, 0.983] |     0.799 |           18 |        66 |
| re-render  | MS-SSIM                           | 0.871 | [0.777, 0.930] |     1.111 |           18 |        66 |
| re-render  | Edge F1                           | 0.854 | [0.807, 0.911] |     1.153 |           18 |        66 |
| re-render  | SSIM (unweighted)                 | 0.758 | [0.694, 0.823] |     0.981 |           18 |        66 |
| re-render  | SSIM (fg-weighted)                | 0.711 | [0.621, 0.785] |     0.862 |           18 |        66 |
| screenshot | Colour histogram                  | 0.912 | [0.873, 0.961] |     0.88  |           36 |        66 |
| screenshot | ΔE00                              | 0.883 | [0.816, 0.946] |     1.253 |           36 |        66 |
| screenshot | PSNR                              | 0.818 | [0.793, 0.875] |     1.182 |           36 |        66 |
| screenshot | Framework score                   | 0.749 | [0.699, 0.803] |     0.956 |           36 |        66 |
| screenshot | Framework composite P (unclipped) | 0.749 | [0.697, 0.807] |     0.956 |           36 |        66 |
| screenshot | Edge F1                           | 0.689 | [0.626, 0.780] |     0.706 |           36 |        66 |
| screenshot | SSIM (unweighted)                 | 0.602 | [0.572, 0.696] |     0.306 |           36 |        66 |
| screenshot | MS-SSIM                           | 0.561 | [0.513, 0.608] |     0.174 |           36 |        66 |
| screenshot | SSIM (fg-weighted)                | 0.447 | [0.412, 0.522] |    -0.19  |           36 |        66 |

## C1. Render-level scores

| pipeline   | kind     | family             |   mean |    std |    min |     max |   count |
|:-----------|:---------|:-------------------|-------:|-------:|-------:|--------:|--------:|
| re-render  | error    | extra_object       | 29.241 | 17.617 | 13.224 |  62.016 |       6 |
| re-render  | error    | hue                | 83.348 | 11.501 | 55.388 |  98.794 |      18 |
| re-render  | error    | lighting           | 62.252 | 24.274 | 18.951 |  92.015 |      18 |
| re-render  | error    | missing_object     | 34.62  | 27.535 | 12.92  |  87.671 |       6 |
| re-render  | error    | proportion         | 55.308 | 21.282 | 18.784 |  90.709 |      18 |
| re-render  | nuisance | samples 16 / 384px | 97.94  |  1.441 | 96.071 | 100     |       6 |
| re-render  | nuisance | samples 64 / 192px | 97.582 |  2.38  | 94.142 | 100     |       6 |
| re-render  | nuisance | samples 8 / 256px  | 96.726 |  2.821 | 92.614 | 100     |       6 |
| screenshot | error    | extra_object       | 24.709 |  7.678 | 12.915 |  33.114 |       6 |
| screenshot | error    | hue                | 33.662 | 12.906 | 10.605 |  55.35  |      18 |
| screenshot | error    | lighting           | 34.333 | 16.113 |  9.125 |  64.419 |      18 |
| screenshot | error    | missing_object     | 24.845 | 10.739 | 14.048 |  40.511 |       6 |
| screenshot | error    | proportion         | 32.05  |  7.369 | 19.032 |  41.685 |      18 |
| screenshot | nuisance | viewpoint          | 43.776 | 13.003 | 18.909 |  69.718 |      36 |

## A1. Image channel — monotonicity

| family      |   rho_level_score |   score_L0 |   score_L0.4 |   score_L1 | monotone_means   |
|:------------|------------------:|-----------:|-------------:|-----------:|:-----------------|
| blur        |            -0.783 |        100 |       82.133 |     64.841 | True             |
| brightness  |            -0.99  |        100 |       74.398 |     38.252 | True             |
| hue         |            -0.781 |        100 |       82.763 |     73.525 | True             |
| jpeg        |            -0.887 |        100 |       99.401 |     61.95  | True             |
| noise       |            -0.893 |        100 |       52.918 |     16.64  | True             |
| occlusion   |            -0.868 |        100 |       97.765 |     79.016 | True             |
| rotation    |            -0.944 |        100 |       55.413 |     34.247 | True             |
| scale       |            -0.892 |        100 |       43.708 |     26.696 | True             |
| translation |            -0.82  |        100 |       34.87  |     26.067 | False            |

## A2. Image channel — scores under nuisance

| family     |    mean |   std |     min |
|:-----------|--------:|------:|--------:|
| n_jpeg     | 100     | 0     | 100     |
| n_noise    |  99.925 | 0.192 |  99.082 |
| n_resample |  99.914 | 0.383 |  98.243 |
| n_shift    |  95.515 | 4.033 |  84.976 |

## B1. Mesh channel — nuisance invariance

| family          |   n |   score_mean |   score_min |   cd_mean |   cd_noalign_mean |   fscore_mean |
|:----------------|----:|-------------:|------------:|----------:|------------------:|--------------:|
| micro_noise     | 144 |          100 |         100 |     0.012 |             0.246 |             1 |
| retriangulation |  48 |          100 |         100 |     0.024 |             0.247 |             1 |
| rigid_rotation  | 144 |          100 |         100 |     0     |             0.253 |             1 |
| uniform_scale   | 144 |          100 |         100 |     0     |             0.246 |             1 |

## B2. Mesh channel — monotonicity

| family            |   rho_level_score |   score_L0 |   score_L0.4 |   score_L1 | monotone_means   |
|:------------------|------------------:|-----------:|-------------:|-----------:|:-----------------|
| anisotropic_scale |            -0.92  |        100 |       64.703 |     10.455 | True             |
| extra_part        |            -0.88  |        100 |       29.419 |      7     | True             |
| hole              |            -0.824 |        100 |       75.822 |     52.782 | True             |
| part_removal      |            -0.824 |        100 |       60.93  |     19.635 | True             |
| vertex_noise      |            -0.844 |        100 |       82.239 |     58.666 | True             |

## A3. Image metric sensitivity |ρ|

|                   |   blur |   brightness |   hue |   jpeg |   noise |   occlusion |   rotation |   scale |   translation |
|:------------------|-------:|-------------:|------:|-------:|--------:|------------:|-----------:|--------:|--------------:|
| Score             |  0.783 |        0.99  | 0.781 |  0.887 |   0.893 |       0.868 |      0.944 |   0.892 |         0.82  |
| PSNR              |  0.959 |        0.991 | 0.951 |  0.889 |   0.89  |       0.889 |      0.776 |   0.741 |         0.752 |
| SSIM (unweighted) |  0.905 |        0.991 | 0.577 |  0.848 |   0.886 |       0.941 |      0.708 |   0.552 |         0.646 |
| SSIM (weighted)   |  0.836 |        0.991 | 0.531 |  0.816 |   0.89  |       0.92  |      0.794 |   0.675 |         0.471 |
| MS-SSIM           |  0.879 |        0.99  | 0.521 |  0.905 |   0.89  |       0.933 |      0.933 |   0.866 |         0.792 |
| ΔE00              |  0.882 |        0.991 | 0.879 |  0.944 |   0.905 |       0.903 |      0.953 |   0.947 |         0.939 |
| Hist. BC          |  0.962 |        0.946 | 0.454 |  0.872 |   0.904 |       0.929 |      0.423 |   0.889 |         0.525 |
| Edge F1           |  0.774 |        0.913 | 0.586 |  0.77  |   0.866 |       0.901 |      0.952 |   0.867 |         0.808 |

## B3. Mesh metric sensitivity |ρ|

|                |   anisotropic_scale |   extra_part |   hole |   part_removal |   vertex_noise |
|:---------------|--------------------:|-------------:|-------:|---------------:|---------------:|
| Score          |               0.92  |        0.88  |  0.824 |          0.824 |          0.844 |
| CD (no align)  |               0.031 |        0.186 |  0.054 |          0.139 |          0.076 |
| CD             |               0.926 |        0.895 |  0.839 |          0.796 |          0.892 |
| HD             |               0.929 |        0.384 |  0.851 |          0.806 |          0.904 |
| H95            |               0.953 |        0.908 |  0.866 |          0.828 |          0.897 |
| F-score        |               0.869 |        0.896 |  0.859 |          0.779 |          0.87  |
| Boundary edges |               0     |        0     |  0.561 |          0.424 |          0     |

## D. Robustness: UI screenshots and auxiliary objects

| test            | method         |    mean |    std |   min |   max |   count |
|:----------------|:---------------|--------:|-------:|------:|------:|--------:|
| UI screenshot   | auto crop      | 100     |  0     |   100 |   100 |      63 |
| UI screenshot   | no crop        |   0     |  0     |     0 |     0 |      63 |
| auxiliary floor | helper removal | 100     |  0     |   100 |   100 |      32 |
| auxiliary floor | no removal     |  74.432 | 43.682 |     0 |   100 |      32 |

Determinism: {'max_abs_pixel_diff': 0, 'identical_obj': True, 'render_seconds': 1.03}

## Runtime (s per assessment)

|   image_s |   mesh_s |   blender_render_s |
|----------:|---------:|-------------------:|
|     0.069 |    0.072 |               1.03 |

## Results paragraph (EN)

In the image channel the framework score separated nuisance variation from errors of severity ≥ 0.2 with AUC = 0.914 ([0.887, 0.938]), compared with 0.919 for PSNR and 0.769 for unweighted SSIM; the strongest single baseline was Edge F1 (AUC = 0.934). The score decreased monotonically with severity for 8 of 9 error families (median Spearman ρ = -0.89). The mesh channel was invariant to rigid rotation, uniform scaling, re-triangulation and micro-noise (mean score 100.0, minimum 100.0; mean Chamfer distance 0.009 versus 0.248 without pose alignment). Against nuisance variation the framework reached AUC = 0.971 for geometric errors, versus 0.541 for Chamfer distance without alignment. For render-level errors in six Blender compositions, standardised re-rendering raised the framework's AUC from 0.749 (screenshots from jittered viewpoints) to 0.985; correct scenes scored 97.4 on average when re-rendered with different sample counts and resolutions, but 43.8 as viewpoint-jittered screenshots. Re-rendering was deterministic (maximum pixel difference 0, identical mesh export: True). Screenshots that contained the application interface scored 0.0 on average and 100.0 after automatic viewport cropping; an auxiliary floor reduced the mesh score to 74.4 without and 100.0 with helper-component removal. Mean assessment time was 0.07 s per image and 0.07 s per mesh on a CPU; one Blender re-render took 1.03 s.
