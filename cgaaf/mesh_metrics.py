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
CD_FLOOR, CD_SIGMA = 0.03, 0.12
H95_FLOOR, H95_SIGMA = 0.06, 0.30
MESH_EXT = {".obj", ".stl", ".ply", ".off", ".glb", ".gltf"}


def drop_plane_components(m: trimesh.Trimesh):
    """Remove plane-like helper components (floors, backdrops). Returns (mesh, n_dropped)."""
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return m, 0
    keep = []
    for p in parts:
        ext = np.sort(p.bounding_box_oriented.extents) if len(p.vertices) >= 4 else np.zeros(3)
        if not (ext[2] > 0 and ext[0] < 0.01 * ext[2] and len(p.faces) <= 128):
            keep.append(p)
    if not keep or len(keep) == len(parts):
        return m, 0
    return trimesh.util.concatenate(keep), len(parts) - len(keep)


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


def topo_score(t: dict) -> float:
    bad = t["boundary_edges"] + 2 * t["nonmanifold_edges"] + t["degenerate_faces"]
    return float(np.clip(1 - bad / (0.05 * max(t["E"], 1)), 0, 1))


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


def prepare_reference(m: trimesh.Trimesh, align="pca_icp", return_scale=False):
    P, s = normalise(sample_points(m))
    P = pca_align(P) if align == "pca_icp" else P
    return (P, float(s)) if return_scale else P


def align_cloud(S, ref, align="pca_icp"):
    if align == "none":
        return S
    if align == "pca_icp":
        S = pca_align(S)
        rs = ref[:: max(1, len(ref) // 512)]
        best = None
        for f in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
            Q = S * np.array(f, float)
            c = chamfer_quick(Q[:: max(1, len(Q) // 512)], rs)
            if best is None or c < best[0]:
                best = (c, Q)
        S = best[1]
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


def compare_meshes(stu_mesh, ref_points: np.ndarray, align="pca_icp", ref_scale=None) -> MeshResult:
    if not isinstance(stu_mesh, trimesh.Trimesh):
        stu_mesh = load_mesh(stu_mesh)
    topo = topology(stu_mesh)
    S0, s_stu = normalise(sample_points(stu_mesh))
    cd_noalign = chamfer_quick(S0, ref_points)  # baseline: no pose alignment
    S = align_cloud(S0, ref_points, align)
    dab = cKDTree(ref_points).query(S)[0]
    dba = cKDTree(S).query(ref_points)[0]
    cd = 0.5 * (dab.mean() + dba.mean())
    hd = max(dab.max(), dba.max())
    h95 = max(np.percentile(dab, 95), np.percentile(dba, 95))
    p, r = (dab < TAU).mean(), (dba < TAU).mean()
    f = 2 * p * r / (p + r) if p + r else 0.0
    raw = {"cd": float(cd), "hd": float(hd), "h95": float(h95), "precision": float(p), "recall": float(r), "fscore": float(f), "cd_noalign": float(cd_noalign), **topo}
    m = {"cd": float(np.exp(-max(0, cd - CD_FLOOR) / CD_SIGMA)), "h95": float(np.exp(-max(0, h95 - H95_FLOOR) / H95_SIGMA)),
         "fscore": float(f), "topo": topo_score(topo)}
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
