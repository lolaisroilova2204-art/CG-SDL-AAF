"""Geometric comparison of a student mesh with a reference mesh.

load (trimesh) -> topology check -> seeded area-weighted surface sampling ->
centre + unit-RMS scaling -> PCA canonical pose (third-moment sign rule) ->
best of 4 proper flips -> ICP (Kabsch/SVD, k-d tree) -> Chamfer, Hausdorff,
H95, F-score@tau.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

N_POINTS = 4096
SEED = 20260101
TAU = 0.1
TOPO_FRAC = 0.05  # defects are tolerated up to 5 % of the edges
CD_FLOOR, CD_SIGMA = 0.03, 0.12
H95_FLOOR, H95_SIGMA = 0.06, 0.30
MESH_EXT = {".obj", ".stl", ".ply", ".off", ".glb", ".gltf"}


def drop_plane_components(m: trimesh.Trimesh):
    """Remove plane-like helper components (floors, backdrops). Returns (mesh, n_dropped).

    A connected component is a helper if it is flat (oriented-bounding-box thickness below 1 % of its
    largest extent) and at least 0.8 times as large as the rest of the model, or if it is a thin slab
    (thickness below 3 %) at least 1.5 times as large as the rest. The size condition keeps small thin
    parts that belong to the model (plates, signs); the face count no longer matters, so subdivided
    floors are recognised as well. The faces of the kept components are taken from the original mesh,
    so vertices shared between parts stay welded and the topology of the model is unchanged."""
    if len(m.faces) == 0:
        return m, 0
    labels = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(m.faces))
    comps = [np.flatnonzero(labels == k) for k in range(int(labels.max()) + 1)]
    if len(comps) <= 1:
        return m, 0
    tri = m.triangles
    boxes, exts = [], []
    for f in comps:
        pts = tri[f].reshape(-1, 3)
        boxes.append((pts.min(0), pts.max(0)))
        sub = trimesh.Trimesh(pts, np.arange(len(pts)).reshape(-1, 3), process=False)
        exts.append(np.sort(sub.bounding_box_oriented.extents) if len(pts) >= 4 else np.zeros(3))
    keep, drop = np.ones(len(m.faces), bool), 0
    for i, f in enumerate(comps):
        t, L = exts[i][0], exts[i][2]
        others = [boxes[k] for k in range(len(comps)) if k != i]
        lo = np.min([b[0] for b in others], axis=0)
        hi = np.max([b[1] for b in others], axis=0)
        E_rest = float(np.max(hi - lo))
        flat = L > 0 and t < 0.01 * L and L >= 0.8 * E_rest
        slab = L > 0 and t < 0.03 * L and L >= 1.5 * E_rest
        if (flat or slab) and E_rest > 0:
            keep[f] = False
            drop += 1
    if drop == 0 or not keep.any():
        return m, 0
    out = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    out.remove_unreferenced_vertices()
    return out, drop


def load_mesh(path, drop_planes=False) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.to_geometry() if hasattr(m, "to_geometry") else m.dump(concatenate=True)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError(f"No triangle mesh found in {Path(path).name}")
    m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
    m.merge_vertices(digits_vertex=6)  # weld duplicated vertices (STL)
    if drop_planes:
        m, _ = drop_plane_components(m)
    return m


def topology(m: trimesh.Trimesh) -> dict:
    e = np.sort(m.edges, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    area = m.area_faces
    eps = (np.abs(m.vertices).max() * 1e-7) ** 2 if len(m.vertices) else 0
    f = m.faces
    degenerate = int(((area <= eps) | (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])).sum())
    labels = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(f))
    b, nm = int((counts == 1).sum()), int((counts > 2).sum())
    return {"V": int(len(m.vertices)), "F": int(len(f)), "E": int(len(counts)), "area": float(area.sum()),
            "boundary_edges": b, "nonmanifold_edges": nm, "degenerate_faces": degenerate,
            "components": int(labels.max() + 1) if len(labels) else 0, "watertight": bool(b == 0 and nm == 0)}


def defect_excess(t: dict, ref: dict | None = None) -> dict:
    """Topological defects of the student mesh in excess of the reference rate (per edge).
    Without a reference the absolute counts are returned (v1.3 behaviour)."""
    out = {}
    tol = max(2.0, 0.005 * t["E"])  # a genuine hole has at least three boundary edges
    for k in ("boundary_edges", "nonmanifold_edges", "degenerate_faces"):
        if ref:
            expect = ref[k] * t["E"] / max(ref["E"], 1)
            out[k] = float(max(0.0, t[k] - expect - tol))
        else:
            out[k] = float(t[k])
    return out


def topo_score(t: dict, ref: dict | None = None) -> float:
    x = defect_excess(t, ref)
    bad = x["boundary_edges"] + 2 * x["nonmanifold_edges"] + x["degenerate_faces"]
    return float(np.clip(1 - bad / (TOPO_FRAC * max(t["E"], 1)), 0, 1))


def d2_histogram(P, bins=64, n_pairs=20000, seed=SEED, r_max=4.0):
    """D2 shape distribution (Osada et al.): histogram of distances between random point pairs of a
    centred, unit-RMS point cloud. Rotation-invariant, needs no alignment."""
    rng = np.random.default_rng(seed + 1)
    i, j = rng.integers(len(P), size=n_pairs), rng.integers(len(P), size=n_pairs)
    h, _ = np.histogram(np.linalg.norm(P[i] - P[j], axis=1), bins=bins, range=(0, r_max))
    return h / max(h.sum(), 1)


def sample_points(m: trimesh.Trimesh, n=N_POINTS, seed=SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tri = m.triangles
    a = m.area_faces
    idx = rng.choice(len(tri), size=n, p=a / a.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    return (tri[idx] * w[:, :, None]).sum(1)


def normalise(P):
    P = P - P.mean(0)
    s = np.sqrt((P ** 2).sum(1).mean()) or 1.0
    return P / s, s


def pca_align(P):
    C = P.T @ P / len(P)
    vals, vecs = np.linalg.eigh(C)
    R = vecs[:, np.argsort(vals)[::-1]].T
    if np.linalg.det(R) < 0:
        R[2] *= -1
    Q = P @ R.T
    sg = np.where((Q ** 3).sum(0) < 0, -1.0, 1.0)
    if np.prod(sg) < 0:
        sg[2] *= -1
    return Q * sg


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def icp(P, tree: cKDTree, target, iters=30):
    R_acc, t_acc, Q = np.eye(3), np.zeros(3), P.copy()
    for _ in range(iters):
        _, j = tree.query(Q)
        R, t = kabsch(Q, target[j])
        Q = Q @ R.T + t
        R_acc, t_acc = R @ R_acc, R @ t_acc + t
        if np.linalg.norm(R - np.eye(3)) < 1e-7 and np.linalg.norm(t) < 1e-7:
            break
    return R_acc, t_acc


def chamfer_quick(A, B):
    return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


def prepare_reference_raw(m: trimesh.Trimesh):
    """Reference cloud centred and scaled to unit RMS radius, in its original orientation."""
    return normalise(sample_points(m))[0]


def prepare_reference(m: trimesh.Trimesh, align="pca_icp", return_scale=False):
    P, s = normalise(sample_points(m))
    P = pca_align(P) if align == "pca_icp" else P
    return (P, float(s)) if return_scale else P


def align_cloud(S, ref, align="pca_icp"):
    """none | pca_icp (default) | pca_only | icp (ICP from the identity pose)."""
    if align == "none":
        return S
    if align in ("pca_icp", "pca_only"):
        S = pca_align(S)
        rs = ref[:: max(1, len(ref) // 512)]
        best = None
        for f in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
            Q = S * np.array(f, float)
            c = chamfer_quick(Q[:: max(1, len(Q) // 512)], rs)
            if best is None or c < best[0]:
                best = (c, Q)
        S = best[1]
        if align == "pca_only":
            return S
    sub = S[:: max(1, len(S) // 2048)]
    R, t = icp(sub, cKDTree(ref), ref)
    return S @ R.T + t


@dataclass
class MeshResult:
    raw: dict
    m: dict
    aligned: np.ndarray = field(repr=False)
    reference: np.ndarray = field(repr=False)


SIZE_SIGMA = 0.10  # |ln(size ratio)| scale: ratio 1.1 -> m_size ~ 0.39


def compare_meshes(stu_mesh, ref_points: np.ndarray, align="pca_icp", ref_scale=None, ref_topo: dict | None = None,
                   baselines: bool = False, ref_raw: np.ndarray | None = None) -> MeshResult:
    """ref_topo: topology(reference mesh); defects are then penalised only in excess of the reference
    (open or non-manifold references are not penalised). baselines=True adds the alignment baselines
    (PCA only, ICP only) and the alignment-free D2 shape-distribution distance to raw.
    ref_raw: the reference cloud centred and scaled but NOT rotated into the PCA frame; the
    no-alignment and ICP-only baselines must be computed against it (v1.3 used the PCA-rotated
    reference, which made even an identical copy look misaligned)."""
    if not isinstance(stu_mesh, trimesh.Trimesh):
        stu_mesh = load_mesh(stu_mesh)
    topo = topology(stu_mesh)
    S0, s_stu = normalise(sample_points(stu_mesh))
    R0 = ref_raw if ref_raw is not None else ref_points
    cd_noalign = chamfer_quick(S0, R0)  # baseline: no pose alignment (original orientations)
    S = align_cloud(S0, ref_points, align)
    dab = cKDTree(ref_points).query(S)[0]
    dba = cKDTree(S).query(ref_points)[0]
    cd = 0.5 * (dab.mean() + dba.mean())
    hd = max(dab.max(), dba.max())
    h95 = max(np.percentile(dab, 95), np.percentile(dba, 95))
    fs = {}
    for tau in (0.05, TAU, 0.15):
        p_, r_ = (dab < tau).mean(), (dba < tau).mean()
        fs[tau] = (float(p_), float(r_), float(2 * p_ * r_ / (p_ + r_)) if p_ + r_ else 0.0)
    p, r, f = fs[TAU]
    exc = defect_excess(topo, ref_topo)
    raw = {"cd": float(cd), "hd": float(hd), "h95": float(h95), "precision": p, "recall": r, "fscore": f,
           "fscore_005": fs[0.05][2], "fscore_015": fs[0.15][2], "cd_noalign": float(cd_noalign), **topo,
           "excess_boundary": exc["boundary_edges"], "excess_nonmanifold": exc["nonmanifold_edges"],
           "excess_degenerate": exc["degenerate_faces"], "topo_relative": bool(ref_topo)}
    if baselines:
        Sp = align_cloud(S0, ref_points, "pca_only")
        Si = align_cloud(S0, R0, "icp")
        raw["cd_pca"] = float(chamfer_quick(Sp, ref_points))
        raw["cd_icp"] = float(chamfer_quick(Si, R0))
        raw["d2"] = float(np.abs(d2_histogram(S0) - d2_histogram(ref_points)).sum())
    m = {"cd": float(np.exp(-max(0, cd - CD_FLOOR) / CD_SIGMA)), "h95": float(np.exp(-max(0, h95 - H95_FLOOR) / H95_SIGMA)),
         "fscore": float(f), "topo": topo_score(topo, ref_topo)}
    raw["rms_radius"] = float(s_stu)
    if ref_scale:
        ratio = float(s_stu / ref_scale)
        raw["size_ratio"] = ratio
        m["size"] = float(np.exp(-abs(np.log(max(ratio, 1e-9))) / SIZE_SIGMA))
    return MeshResult(raw, m, S, ref_points)


# ---------------------------------------------------------------- synthetic shapes (demo/tests)
def torus(R=1.0, r=0.35, nu=48, nv=24, rot=0.0, scale=1.0, noise=0.0, hole=False, seed=7) -> trimesh.Trimesh:
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, nu, endpoint=False), np.linspace(0, 2 * np.pi, nv, endpoint=False), indexing="ij")
    rr = r * (1 + noise * (rng.random(u.shape) - 0.5))
    x, y, z = (R + rr * np.cos(v)) * np.cos(u), (R + rr * np.cos(v)) * np.sin(u), rr * np.sin(v)
    c, s = np.cos(rot), np.sin(rot)
    V = np.stack([x, y * c - z * s, y * s + z * c], -1).reshape(-1, 3) * scale
    F = []
    for i in range(nu):
        if hole and i < 3:
            continue
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            cc, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            F += [[a, b, cc], [a, cc, d]]
    return trimesh.Trimesh(V, np.array(F), process=False)
