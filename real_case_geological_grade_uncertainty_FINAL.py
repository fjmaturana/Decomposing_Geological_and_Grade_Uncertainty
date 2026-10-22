"""
Geological versus grade uncertainty in the Monty V_UZ1 operational case.

The primary geological experiment perturbs boundary position on the actual
triangulated reference surface while holding mineralised volume and domain
density fixed. Geological fields are independent outer Monte Carlo draws.
Support already assigned to another named wireframe is ineligible to enter
V_UZ1 during outward perturbation.

Required files in this script's folder
--------------------------------------
mod_base.csv
mnc.csv
v_uz1.dxf

Required packages
-----------------
numpy, pandas, scipy, matplotlib, ezdxf

Methodological safeguards
-------------------------
1. Exact point-to-triangle distances are used rather than distances to DXF
   vertices.
2. The actual closed DXF triangular surface is used.
3. Coarse cells in the boundary shell are subdivided to approximately 1 m
   support before geological alternatives are evaluated.
4. The fine reference support is reconciled to the deterministic V_UZ1 block
   model before stochastic results are accepted.
5. Boundary fields are generated on the reference surface and evaluated at
   closest-surface locations.
6. The geological realisations are independent, matching the independence
   assumption used by the nested variance estimator.
7. The volume-constrained selector enforces the V_UZ1 eligibility mask, so
   material belonging to another named wireframe cannot be recruited.
8. Directional log-Cu variograms use an exponential model. Random Fourier
   features use the matching isotropic exponential covariance after
   anisotropic coordinate scaling.
9. Grade simulations remain conditional on the operational composites.
10. Boundary SD and boundary correlation range remain sensitivity parameters,
    not empirically measured geological uncertainty.

The script exports real_case_results.tex as an audit output. The submission
manuscript contains its numerical results directly and does not require this
generated file for compilation. Rerun this script and reconcile the exported
results with the manuscript before publication.
"""

from pathlib import Path
import json
import hashlib
import math
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Springer Nature artwork settings. DejaVu Sans is used explicitly because
# it embeds reliably in EPS/PDF output across Matplotlib, Ghostscript and
# pdfLaTeX workflows. Lettering is sized to remain legible after manuscript
# scaling.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11.0,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "lines.linewidth": 1.1,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

from scipy.spatial import cKDTree
from scipy.optimize import least_squares

import ezdxf


# ======================================================================
# CONFIGURATION
# ======================================================================

ROOT = Path(__file__).resolve().parent

BLOCK_CANDIDATES = [
    ROOT / "mod_base.csv",
    ROOT / "mod_base(1).csv",
]
COMP_FILE = ROOT / "mnc.csv"
DXF_FILE = ROOT / "v_uz1.dxf"

OUT = ROOT / "real_case_uncertainty_outputs_final"
OUT.mkdir(exist_ok=True)

SEED = 20260925

TARGET_WF = "V_UZ1"
TARGET_COMP_DOMAIN = 32

N_GEO = 50
N_GRADE = 50

BOUNDARY_SDS_M = (1.0, 2.0, 5.0)
BOUNDARY_RANGE_M = 40.0

# Shell must exceed the largest displacement scenario.
SHELL_M = 25.0

# Fine integration support. 1 m is appropriate for 1–5 m perturbations.
FINE_SUPPORT_M = 1.0

# A native cell is subdivided if any dimension is larger than this.
SUBDIVIDE_IF_DIM_GT_M = 1.25

# To control memory, candidate blocks whose centres are far outside the
# reference surface are excluded before fine subdivision.
COARSE_PREFILTER_M = SHELL_M + 10.0

CUTOFF_CU_PCT = 0.0

N_GEO_FEATURES = 128
N_GRADE_FEATURES = 180
LOGCU_NOISE_SD = 0.12

GRADE_CAP_LOW_Q = 0.001
GRADE_CAP_HIGH_Q = 0.999

# Triangle-centroid KD-tree broadphase. If neither the first 16 nor the
# expanded 64 candidates can be certified, a guaranteed exhaustive-radius
# fallback tests all geometrically possible closer triangles. The 64-candidate
# setting therefore limits the fast search only, not correctness.
TRIANGLE_K_INITIAL = 16
TRIANGLE_K_MAX = 64

POINT_BATCH = 20000
FEATURE_BATCH = 25000

# Reference reconciliation tolerance after integration support is built.
REFERENCE_TONNAGE_REL_TOL = 0.01

# Constrained grade range bounds as fractions of observed PCA extent.
RANGE_MIN_M = np.array([10.0, 3.0, 2.0])
RANGE_MAX_FRACTION_OF_EXTENT = np.array([0.75, 0.75, 0.75])


# ======================================================================
# INPUT CHECKS
# ======================================================================

def find_existing(candidates, label):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{label} not found. Expected one of:\n"
        + "\n".join(str(x) for x in candidates)
    )


BLOCK_FILE = find_existing(BLOCK_CANDIDATES, "Block model")

for p, label in [(COMP_FILE, "Composite file"), (DXF_FILE, "DXF")]:
    if not p.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{p}\n"
            f"Place it in the same folder as this script."
        )

print("INPUT FILES")
print("  Block model :", BLOCK_FILE)
print("  Composites  :", COMP_FILE)
print("  Wireframe   :", DXF_FILE)
print("  Output      :", OUT)
print()

# Each full analysis run uses a cache namespace tied to the *contents* of the
# three operational inputs, this source file and the relevant environment.
# Changing any input, model setting or code cannot silently reuse old results.
# The cache is kept inside the usual output directory for Spyder button-runs.
def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


RUN_PROVENANCE = {
    "script_sha256": sha256_file(Path(__file__).resolve()),
    "input_files": {
        label: {"filename": path.name, "sha256": sha256_file(path)}
        for label, path in (
            ("block_model", BLOCK_FILE),
            ("composites", COMP_FILE),
            ("wireframe", DXF_FILE),
        )
    },
    "environment": {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "ezdxf": getattr(ezdxf, "__version__", "unknown"),
    },
    "model_settings": {
        "seed": SEED,
        "target_wireframe": TARGET_WF,
        "target_composite_domain": TARGET_COMP_DOMAIN,
        "n_geological_realisations": N_GEO,
        "n_grade_realisations": N_GRADE,
        "boundary_sds_m": list(BOUNDARY_SDS_M),
        "boundary_range_m": BOUNDARY_RANGE_M,
        "shell_m": SHELL_M,
        "fine_support_m": FINE_SUPPORT_M,
        "subdivide_if_dim_gt_m": SUBDIVIDE_IF_DIM_GT_M,
        "coarse_prefilter_m": COARSE_PREFILTER_M,
        "cutoff_cu_pct": CUTOFF_CU_PCT,
        "n_geo_features": N_GEO_FEATURES,
        "n_grade_features": N_GRADE_FEATURES,
        "logcu_noise_sd": LOGCU_NOISE_SD,
        "grade_cap_low_q": GRADE_CAP_LOW_Q,
        "grade_cap_high_q": GRADE_CAP_HIGH_Q,
        "range_min_m": RANGE_MIN_M.tolist(),
        "range_max_fraction_of_extent": RANGE_MAX_FRACTION_OF_EXTENT.tolist(),
        "triangle_k_initial": TRIANGLE_K_INITIAL,
        "triangle_k_max": TRIANGLE_K_MAX,
    },
}
RUN_SIGNATURE = hashlib.sha256(
    json.dumps(RUN_PROVENANCE, sort_keys=True).encode("utf-8")
).hexdigest()
RUN_CACHE = OUT / "run_cache" / RUN_SIGNATURE[:20]
RUN_CACHE.mkdir(parents=True, exist_ok=True)
(RUN_CACHE / "run_provenance.json").write_text(
    json.dumps({**RUN_PROVENANCE, "run_signature": RUN_SIGNATURE}, indent=2),
    encoding="utf-8",
)
print("RUN PROVENANCE")
print("  Signature :", RUN_SIGNATURE[:20])
print("  Cache     :", RUN_CACHE)
print("  Previous incompatible checkpoints will not be reused.")
print()


# ======================================================================
# GENERAL HELPERS
# ======================================================================

def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    q = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[q] * w[q]) / np.sum(w[q]))


def pca_axes(xyz):
    centre = xyz.mean(axis=0)
    X = xyz - centre
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
    order = np.argsort(vals)[::-1]
    return centre, vals[order], vecs[:, order]


# ======================================================================
# DXF TRIANGLES
# ======================================================================

def load_dxf_triangles(path):
    doc = ezdxf.readfile(path)
    triangles = []

    for e in doc.modelspace().query("3DFACE"):
        a = np.asarray(e.dxf.vtx0, dtype=float)
        b = np.asarray(e.dxf.vtx1, dtype=float)
        c = np.asarray(e.dxf.vtx2, dtype=float)

        # Reject degenerate triangles.
        if np.linalg.norm(np.cross(b - a, c - a)) > 1e-10:
            triangles.append([a, b, c])

    triangles = np.asarray(triangles, dtype=float)

    if len(triangles) == 0:
        raise ValueError("No valid 3DFACE triangles found in DXF.")

    return triangles


def closest_point_on_triangle(p, tri):
    """
    Exact closest point on triangle to point p.

    Uses the standard Voronoi-region tests from Ericson. Returning the
    closest surface point is essential in because geological displacement
    is generated on the reference contact and then extended only along local
    surface normals.
    """
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def point_triangle_distance_sq(p, tri):
    q = closest_point_on_triangle(p, tri)
    d = p - q
    return float(np.dot(d, d))

def exact_surface_distance(points, triangles, batch=POINT_BATCH,
                           k_initial=TRIANGLE_K_INITIAL,
                           k_max=TRIANGLE_K_MAX):
    """Globally certified closest point and distance to a triangle mesh.

    The triangle-centroid KD tree supplies an initial shortlist. Let c_i be a
    triangle centroid and r_i the greatest vertex-to-centroid distance.
    Every point q in that triangle satisfies ||q-c_i|| <= r_i, so an omitted
    triangle with centroid distance d cannot beat the current best b if
    d-r_i > b. Using the global maximum r_i certifies the shortlist when
    kth_centroid_distance - max_radius > b.

    If the first 16 and the expanded 64 candidates cannot be certified, we
    query *all* centroids within b + max_radius. No omitted triangle can be
    closer than b. Candidate-specific bounding spheres and axis-aligned
    triangle boxes further prune this certified fallback. There is no fixed
    maximum candidate count in the final search.

    Returns exactly the nearest triangle point to floating-point precision,
    even when the closest triangle's centroid is not among the nearest 64.
    This may take longer than the old approximation where fallback is needed.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (n, 3, 3)")
    if len(triangles) == 0:
        raise ValueError("No triangles supplied")
    if not np.isfinite(points).all() or not np.isfinite(triangles).all():
        raise ValueError("Non-finite mesh or point coordinates")

    centroids = triangles.mean(axis=1)
    radii = np.max(np.linalg.norm(triangles - centroids[:, None, :], axis=2),
                   axis=1)
    max_rad = float(np.max(radii))
    aabb_min = triangles.min(axis=1)
    aabb_max = triangles.max(axis=1)
    tree = cKDTree(centroids)
    n_tri = len(triangles)
    n_initial = min(max(1, int(k_initial)), n_tri)
    n_expanded = min(max(n_initial, int(k_max)), n_tri)
    out = np.empty(len(points), dtype=np.float64)
    # Keep the returned closest coordinates at full precision. Near the
    # 71,000 m operational northings, float32 can round by several mm.
    closest = np.empty((len(points), 3), dtype=np.float64)
    fallback_points = 0
    fallback_triangles = 0

    for start in range(0, len(points), batch):
        stop = min(start + batch, len(points))
        P = points[start:stop]
        dc, idx = tree.query(P, k=n_initial, workers=-1)
        if n_initial == 1:
            dc = dc[:, None]
            idx = idx[:, None]

        for local_i, p in enumerate(P):
            visited = set()
            best2 = np.inf
            bestq = None

            def examine(tri_id):
                nonlocal best2, bestq
                tri_id = int(tri_id)
                if tri_id in visited:
                    return
                visited.add(tri_id)
                q = closest_point_on_triangle(p, triangles[tri_id])
                d2 = float(np.dot(p - q, p - q))
                if d2 < best2:
                    best2, bestq = d2, q

            for tri_id in np.atleast_1d(idx[local_i]):
                examine(tri_id)

            best = math.sqrt(best2)
            # Tolerance handles nearly tangent/degenerate equal distances.
            tol = 1e-10 * max(1.0, best, max_rad)
            kth_dist = float(np.max(np.atleast_1d(dc[local_i])))
            certified = (n_initial == n_tri or
                         kth_dist - max_rad > best + tol)

            if not certified and n_expanded > n_initial:
                dc2, ids2 = tree.query(p, k=n_expanded)
                for tri_id in np.atleast_1d(ids2):
                    examine(tri_id)
                best = math.sqrt(best2)
                tol = 1e-10 * max(1.0, best, max_rad)
                kth_dist = float(np.max(np.atleast_1d(dc2)))
                certified = (n_expanded == n_tri or
                             kth_dist - max_rad > best + tol)

            if not certified:
                fallback_points += 1
                # Every potentially closer triangle has its centroid within
                # best + its own radius <= best + max_rad.
                ids_ball = np.asarray(
                    tree.query_ball_point(p, best + max_rad + tol),
                    dtype=np.int64,
                )
                if len(ids_ball) == 0:
                    raise RuntimeError("Certified radius search found no faces")

                # The bounding-sphere and bounding-box lower bounds can only
                # discard a triangle when it provably cannot improve 'best'.
                cd = np.linalg.norm(centroids[ids_ball] - p, axis=1)
                ids_ball = ids_ball[cd - radii[ids_ball] <= best + tol]
                lower_xyz = np.maximum(
                    np.maximum(aabb_min[ids_ball] - p, p - aabb_max[ids_ball]),
                    0.0,
                )
                bb_lower2 = np.einsum("ij,ij->i", lower_xyz, lower_xyz)
                ids_ball = ids_ball[bb_lower2 <= (best + tol) ** 2]
                fallback_triangles += len(ids_ball)
                for tri_id in ids_ball:
                    examine(tri_id)
                # The KD-tree ball query makes this a global certificate.

            if bestq is None:
                raise RuntimeError("Unable to find nearest surface point")
            out[start + local_i] = math.sqrt(best2)
            closest[start + local_i] = bestq

        print(f"  certified surface distance: {stop:,}/{len(points):,}",
              end="\r")

    print()
    print(f"  global-certification fallbacks: {fallback_points:,} points")
    print(f"  triangles in fallback candidate sets: "
          f"{fallback_triangles:,}")
    return out, closest


# ======================================================================
# CLOSED-MESH GEOMETRY
# ======================================================================

def indexed_mesh_from_triangles(triangles, decimals=6):
    flat = np.asarray(triangles, float).reshape(-1, 3)
    key = np.round(flat, decimals=decimals)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    vertices = np.zeros_like(uniq, dtype=float)
    counts = np.zeros(len(uniq), dtype=int)
    for i, j in enumerate(inverse):
        vertices[j] += flat[i]
        counts[j] += 1
    vertices /= counts[:, None]
    return vertices, inverse.reshape(-1, 3).astype(np.int64)


def orient_closed_mesh(vertices, faces):
    faces = np.asarray(faces, np.int64).copy()
    edge_map = {}
    for fi, f in enumerate(faces):
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            key = (min(a, b), max(a, b))
            direction = 1 if a < b else -1
            edge_map.setdefault(key, []).append((fi, direction))

    bad = [e for e, inc in edge_map.items() if len(inc) != 2]
    if bad:
        raise RuntimeError(
            f"DXF mesh is not closed/manifold: {len(bad)} edges "
            "do not have exactly two incident faces."
        )

    adj = [[] for _ in range(len(faces))]
    for inc in edge_map.values():
        (f1, d1), (f2, d2) = inc
        parity = 1 if d1 == d2 else 0
        adj[f1].append((f2, parity))
        adj[f2].append((f1, parity))

    flip = np.full(len(faces), -1, dtype=np.int8)
    components = []
    for root in range(len(faces)):
        if flip[root] >= 0:
            continue
        flip[root] = 0
        stack, comp = [root], []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j, parity in adj[i]:
                wanted = flip[i] ^ parity
                if flip[j] < 0:
                    flip[j] = wanted
                    stack.append(j)
                elif flip[j] != wanted:
                    raise RuntimeError("Inconsistent mesh orientation.")
        components.append(np.asarray(comp, int))

    for i in np.flatnonzero(flip == 1):
        faces[i, [1, 2]] = faces[i, [2, 1]]

    for comp in components:
        tri = vertices[faces[comp]]
        vol6 = np.sum(np.einsum(
            "ij,ij->i", tri[:, 0],
            np.cross(tri[:, 1], tri[:, 2])
        ))
        if vol6 < 0:
            ff = faces[comp].copy()
            ff[:, [1, 2]] = ff[:, [2, 1]]
            faces[comp] = ff
    return faces, components


def vertex_normals(vertices, faces):
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = np.zeros_like(vertices, dtype=float)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1)
    if np.any(ln <= 1e-12):
        raise RuntimeError("Zero-length vertex normal encountered.")
    return n / ln[:, None]


def mesh_signed_volume(vertices, faces):
    tri = vertices[faces]
    return float(np.sum(np.einsum(
        "ij,ij->i", tri[:, 0],
        np.cross(tri[:, 1], tri[:, 2])
    )) / 6.0)


def mesh_min_triangle_area(vertices, faces):
    tri = vertices[faces]
    return float(0.5 * np.min(np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1
    )))


def point_in_closed_mesh(points, vertices, faces, grid_n=96):
    """
    Accelerated odd/even point-in-closed-mesh classification.

    The ray/triangle test uses an exact
    2-D uniform-grid index is built in the plane perpendicular to the ray.
    A point is therefore tested only against triangles whose projected
    bounding boxes overlap its grid cell. This removes the dominant
    all-points x all-triangles cost.

    The grid is rebuilt for each displaced mesh, which is inexpensive for
    ~18k triangles and preserves the actual displaced geometry.
    """
    P = np.asarray(points, dtype=np.float64)
    if len(P) == 0:
        return np.zeros(0, dtype=bool)

    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    tri = V[F]

    ray = np.array([1.0, 0.3713906763541037, 0.217983498771312],
                   dtype=np.float64)
    ray /= np.linalg.norm(ray)

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, ray)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    uax = np.cross(ray, helper)
    uax /= np.linalg.norm(uax)
    vax = np.cross(ray, uax)

    # Project triangles and points to the plane perpendicular to the ray.
    tu = tri @ uax
    tv = tri @ vax
    pu = P @ uax
    pv = P @ vax

    umin = tu.min(axis=1)
    umax = tu.max(axis=1)
    vmin = tv.min(axis=1)
    vmax = tv.max(axis=1)

    gu0 = min(float(umin.min()), float(pu.min()))
    gu1 = max(float(umax.max()), float(pu.max()))
    gv0 = min(float(vmin.min()), float(pv.min()))
    gv1 = max(float(vmax.max()), float(pv.max()))
    du = max((gu1 - gu0) / grid_n, 1e-9)
    dv = max((gv1 - gv0) / grid_n, 1e-9)

    def cell_u(x):
        return np.clip(((x - gu0) / du).astype(np.int32), 0, grid_n - 1)

    def cell_v(x):
        return np.clip(((x - gv0) / dv).astype(np.int32), 0, grid_n - 1)

    # Triangle lists for projected grid cells. A triangle is inserted into
    # every cell touched by its projected bounding box, so candidate
    # rejection is conservative and does not omit a valid ray intersection.
    tri_cells = [[] for _ in range(grid_n * grid_n)]
    iu0, iu1 = cell_u(umin), cell_u(umax)
    iv0, iv1 = cell_v(vmin), cell_v(vmax)

    for ti in range(len(tri)):
        for iv in range(iv0[ti], iv1[ti] + 1):
            base = iv * grid_n
            for iu in range(iu0[ti], iu1[ti] + 1):
                tri_cells[base + iu].append(ti)

    piu = cell_u(pu)
    piv = cell_v(pv)
    pcell = piv * grid_n + piu
    order = np.argsort(pcell, kind="stable")
    sorted_cells = pcell[order]

    v0 = tri[:, 0]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    h = np.cross(np.broadcast_to(ray, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, h)
    valid_tri = np.abs(det) > 1e-12
    inv_det = np.zeros_like(det)
    inv_det[valid_tri] = 1.0 / det[valid_tri]

    inside = np.zeros(len(P), dtype=bool)
    eps = 1e-10

    cuts = np.r_[0, np.flatnonzero(np.diff(sorted_cells)) + 1, len(P)]
    for kk in range(len(cuts) - 1):
        lo, hi = cuts[kk], cuts[kk + 1]
        ids_p = order[lo:hi]
        cid = int(sorted_cells[lo])
        ids_t = np.asarray(tri_cells[cid], dtype=np.int64)

        if len(ids_t) == 0:
            continue
        ids_t = ids_t[valid_tri[ids_t]]
        if len(ids_t) == 0:
            continue

        Q = P[ids_p]
        qu = pu[ids_p]
        qv = pv[ids_p]

        # Exact projected-bounding-box rejection within the cell.
        bbox = (
            (qu[:, None] >= umin[ids_t][None, :] - eps)
            & (qu[:, None] <= umax[ids_t][None, :] + eps)
            & (qv[:, None] >= vmin[ids_t][None, :] - eps)
            & (qv[:, None] <= vmax[ids_t][None, :] + eps)
        )

        sv = Q[:, None, :] - v0[ids_t][None, :, :]
        uu = np.einsum("mti,ti->mt", sv, h[ids_t], optimize=True)
        uu *= inv_det[ids_t][None, :]

        qvec = np.cross(sv, e1[ids_t][None, :, :])
        vv = np.einsum("i,mti->mt", ray, qvec, optimize=True)
        vv *= inv_det[ids_t][None, :]

        tt = np.einsum("ti,mti->mt", e2[ids_t], qvec, optimize=True)
        tt *= inv_det[ids_t][None, :]

        hit = (
            bbox
            & (uu >= -eps)
            & (vv >= -eps)
            & ((uu + vv) < 1.0 - eps)
            & (tt > 1e-9)
        )
        inside[ids_p] = (hit.sum(axis=1) % 2) == 1

    return inside


# ======================================================================
# FINE INTEGRATION SUPPORT
# ======================================================================

def subdivision_centres(xc, yc, zc, dx, dy, dz, target):
    """
    Return regular subcell centres and common subcell volume.

    Number of divisions is ceil(parent dimension / target), so no subcell
    exceeds target in a subdivided dimension.
    """
    nx = max(1, int(math.ceil(dx / target)))
    ny = max(1, int(math.ceil(dy / target)))
    nz = max(1, int(math.ceil(dz / target)))

    sx = dx / nx
    sy = dy / ny
    sz = dz / nz

    xs = xc - dx / 2 + sx * (0.5 + np.arange(nx))
    ys = yc - dy / 2 + sy * (0.5 + np.arange(ny))
    zs = zc - dz / 2 + sz * (0.5 + np.arange(nz))

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    return pts, sx * sy * sz


def build_fine_support(cand):
    """
    Fine support for geometry integration.

    Native cells already near 1 m support are retained. Coarse cells in the
    boundary shell are subdivided. Density is inherited from the parent.
    """
    xyz_list = []
    vol_list = []
    den_list = []
    ref_parent_list = []
    eligible_list = []
    parent_index_list = []

    n = len(cand)

    for counter, (idx, row) in enumerate(cand.iterrows(), start=1):
        dx = float(row.XINC)
        dy = float(row.YINC)
        dz = float(row.ZINC)

        if max(dx, dy, dz) > SUBDIVIDE_IF_DIM_GT_M:
            pts, subv = subdivision_centres(
                float(row.XC), float(row.YC), float(row.ZC),
                dx, dy, dz, FINE_SUPPORT_M
            )
        else:
            pts = np.array([[row.XC, row.YC, row.ZC]], dtype=float)
            subv = dx * dy * dz

        m = len(pts)

        xyz_list.append(pts)
        vol_list.append(np.full(m, subv, dtype=float))
        den_list.append(np.full(m, float(row.DENSITY), dtype=float))
        ref_parent_list.append(
            np.full(m, bool(row.reference_inside), dtype=bool)
        )
        eligible_list.append(
            np.full(m, bool(row.eligible_for_V_UZ1), dtype=bool)
        )
        parent_index_list.append(
            np.full(m, idx, dtype=np.int64)
        )

        if counter % 5000 == 0 or counter == n:
            print(
                f"  support subdivision: {counter:,}/{n:,} parent cells",
                end="\r",
            )

    print()

    return {
        "xyz": np.vstack(xyz_list),
        "volume": np.concatenate(vol_list),
        "density": np.concatenate(den_list),
        "reference_parent": np.concatenate(ref_parent_list),
        "eligible": np.concatenate(eligible_list),
        "parent_index": np.concatenate(parent_index_list),
    }


# ======================================================================
# VARIOGRAM + RANDOM FEATURES
# ======================================================================

def directional_variogram(pc, values, axis, tol_ratio=0.60,
                          nbins=12, max_pairs=250000, seed=123):
    rng = np.random.default_rng(seed + axis)
    n = len(values)
    ii, jj = np.triu_indices(n, 1)

    if len(ii) > max_pairs:
        take = rng.choice(len(ii), max_pairs, replace=False)
        ii = ii[take]
        jj = jj[take]

    d = pc[jj] - pc[ii]
    along = np.abs(d[:, axis])
    cross = np.sqrt(
        np.maximum(np.sum(d * d, axis=1) - d[:, axis] ** 2, 0.0)
    )

    keep = (along > 0) & (cross <= tol_ratio * along)
    h = along[keep]
    gamma = 0.5 * (values[jj[keep]] - values[ii[keep]]) ** 2

    if len(h) == 0:
        return pd.DataFrame(
            columns=["lag_m", "semivariance", "npairs"]
        )

    hmax = np.quantile(h, 0.90)
    edges = np.linspace(0.0, hmax, nbins + 1)

    rows = []
    for b in range(nbins):
        q = (h >= edges[b]) & (h < edges[b + 1])
        if q.sum() >= 8:
            rows.append({
                "lag_m": float(h[q].mean()),
                "semivariance": float(gamma[q].mean()),
                "npairs": int(q.sum()),
            })

    return pd.DataFrame(rows)


def constrained_exp_fit(vg, variance, min_range, max_range):
    """
    Fit exponential variogram under explicit defensible bounds.

    This avoids reporting pathological multi-million-metre ranges when an
    empirical directional variogram does not reach a sill.
    """
    if len(vg) < 4:
        return {
            "nugget": 0.10 * variance,
            "partial_sill": 0.90 * variance,
            "sill": variance,
            "range_parameter": max(min_range, min(max_range, 0.5 * max_range)),
            "fit_status": "fallback_sparse_direction",
        }

    h = vg.lag_m.to_numpy(float)
    y = vg.semivariance.to_numpy(float)
    wt = np.sqrt(vg.npairs.to_numpy(float))
    wt /= wt.max()

    var = max(float(variance), 1e-6)

    # Parameterisation: nugget, partial sill, range.
    def residual(theta):
        nug, psill, arange = theta
        pred = nug + psill * (1.0 - np.exp(-h / arange))
        return wt * (pred - y)

    x0 = np.array([
        min(0.15 * var, max(y.min(), 1e-4)),
        max(0.85 * var, 1e-4),
        np.clip(np.median(h), min_range, max_range),
    ])

    lower = np.array([0.0, 1e-6, min_range])
    upper = np.array([
        2.0 * var,
        3.0 * var,
        max_range,
    ])

    fit = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
    )

    nug, psill, arange = fit.x

    hit_range_bound = (
        abs(arange - min_range) < 1e-4
        or abs(arange - max_range) < 1e-4
    )

    return {
        "nugget": float(nug),
        "partial_sill": float(psill),
        "sill": float(nug + psill),
        "range_parameter": float(arange),
        "fit_status": (
            "range_bound_active"
            if hit_range_bound
            else "constrained_fit"
        ),
    }


def make_exponential_rff(n_features, n_dim, rng):
    """Random Fourier frequencies for k(h)=exp(-||h||).

    The isotropic exponential covariance is the Matérn covariance with
    smoothness 1/2. Its spectral distribution is multivariate Cauchy. A
    multivariate Cauchy vector is generated as Z/|U| with a standard normal
    vector Z and an independent scalar standard normal U. The shared
    denominator is essential. Independent univariate Cauchy components would
    represent a different, separable kernel.
    """
    numerator = rng.normal(size=(n_features, n_dim))
    denominator = np.abs(rng.normal(size=(n_features, 1)))
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    omega = numerator / denominator
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return omega, phase


def rff_matrix(x_scaled, omega, phase):
    return np.sqrt(2.0 / omega.shape[0]) * np.cos(
        x_scaled @ omega.T + phase
    )


def posterior_weights(Phi, y, noise_sd):
    nv = noise_sd ** 2
    A = np.eye(Phi.shape[1]) + Phi.T @ Phi / nv
    L = np.linalg.cholesky(A)
    rhs = Phi.T @ y / nv
    mu = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
    return mu, L


def draw_weights(mu, L, n, rng):
    z = rng.normal(size=(len(mu), n))
    return mu[:, None] + np.linalg.solve(L.T, z)


def decompose_nested(mat):
    mat = np.asarray(mat, float)
    good = np.all(np.isfinite(mat), axis=1)
    mat = mat[good]

    G, R = mat.shape

    within = mat.var(axis=1, ddof=1)
    means = mat.mean(axis=1)

    u_grade = float(within.mean())
    between = float(means.var(ddof=1))
    u_geo_raw = float(between - u_grade / R)
    u_geo = max(0.0, u_geo_raw)
    u_total = u_grade + u_geo
    rg = u_geo / u_total if u_total > 0 else np.nan

    flat = mat.ravel()

    return {
        "G": G,
        "R": R,
        "U_grade": u_grade,
        "S_between_means": between,
        "U_geo_raw": u_geo_raw,
        "U_geo_report": u_geo,
        "U_total_report": u_total,
        "R_G": rg,
        "mean": float(flat.mean()),
        "sd_nested": float(flat.std(ddof=1)),
        "p05": float(np.quantile(flat, 0.05)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
    }


# ======================================================================
# LOAD
# ======================================================================

print("Loading data ...")

blocks = pd.read_csv(BLOCK_FILE, low_memory=False)
comps = pd.read_csv(COMP_FILE, low_memory=False)
triangles = load_dxf_triangles(DXF_FILE)

required_b = [
    "XC", "YC", "ZC", "XINC", "YINC", "ZINC",
    "DENSITY", "CU", "WF"
]
required_c = [
    "X", "Y", "Z", "CU_PCT", "LENGTH", "DOMAIN"
]

for c in required_b:
    if c not in blocks.columns:
        raise ValueError(f"Missing block column: {c}")

for c in required_c:
    if c not in comps.columns:
        raise ValueError(f"Missing composite column: {c}")

for c in ["XC", "YC", "ZC", "XINC", "YINC", "ZINC",
          "DENSITY", "CU"]:
    blocks[c] = pd.to_numeric(blocks[c], errors="coerce")

for c in ["X", "Y", "Z", "CU_PCT", "LENGTH", "DOMAIN"]:
    comps[c] = pd.to_numeric(comps[c], errors="coerce")

print(f"  Blocks       : {len(blocks):,}")
print(f"  Composites   : {len(comps):,}")
print(f"  DXF triangles: {len(triangles):,}")
mesh_vertices, mesh_faces = indexed_mesh_from_triangles(triangles)
mesh_faces, mesh_components = orient_closed_mesh(mesh_vertices, mesh_faces)
mesh_normals = vertex_normals(mesh_vertices, mesh_faces)
reference_mesh_volume = abs(mesh_signed_volume(mesh_vertices, mesh_faces))
print(f"  Indexed vertices: {len(mesh_vertices):,}")
print(f"  Closed components: {len(mesh_components):,}")
print(f"  Mesh volume: {reference_mesh_volume:,.3f} m3")

print()


# ======================================================================
# DETERMINISTIC REFERENCE
# ======================================================================

ref_mask = blocks.WF.fillna("").astype(str).eq(TARGET_WF)
ref = blocks.loc[ref_mask].copy()

if len(ref) == 0:
    raise ValueError(f"No blocks found for WF={TARGET_WF}")

ref_volume_by_block = (
    ref.XINC.to_numpy(float)
    * ref.YINC.to_numpy(float)
    * ref.ZINC.to_numpy(float)
)

ref_density = ref.DENSITY.to_numpy(float)
density_fallback = float(np.nanmedian(ref_density))

ref_density = np.where(
    np.isfinite(ref_density),
    ref_density,
    density_fallback,
)

ref_tonnes_by_block = ref_volume_by_block * ref_density

REF_VOLUME = float(ref_volume_by_block.sum())
REF_TONNES = float(ref_tonnes_by_block.sum())
REF_GRADE = weighted_mean(
    ref.CU.to_numpy(float),
    ref_tonnes_by_block,
)
REF_METAL = float(
    np.sum(
        ref_tonnes_by_block
        * ref.CU.to_numpy(float)
        / 100.0
    )
)

print("DETERMINISTIC V_UZ1")
print(f"  blocks       : {len(ref):,}")
print(f"  volume       : {REF_VOLUME:,.3f} m3")
print(f"  tonnes       : {REF_TONNES:,.3f} t")
print(f"  Cu           : {REF_GRADE:.4f} %")
print(f"  contained Cu : {REF_METAL:,.3f} t")
print()


# ======================================================================
# COMPOSITES
# ======================================================================

gc = comps.loc[
    (comps.DOMAIN == TARGET_COMP_DOMAIN)
    & np.isfinite(comps.CU_PCT)
    & (comps.CU_PCT > 0)
    & np.isfinite(comps.X)
    & np.isfinite(comps.Y)
    & np.isfinite(comps.Z)
].copy()

if len(gc) < 20:
    raise ValueError(
        f"Only {len(gc)} composites in DOMAIN {TARGET_COMP_DOMAIN}."
    )

gc.LENGTH = pd.to_numeric(gc.LENGTH, errors="coerce")
gc.LENGTH = gc.LENGTH.fillna(gc.LENGTH.median())
gc.loc[gc.LENGTH <= 0, "LENGTH"] = gc.LENGTH.median()

COMP_MEAN = weighted_mean(gc.CU_PCT, gc.LENGTH)

# primary mineralised-domain density, calibrated from V_UZ1.
DOMAIN_DENSITY_T_M3 = REF_TONNES / REF_VOLUME
print("PRIMARY DOMAIN DENSITY")
print(f"  calibrated density: {DOMAIN_DENSITY_T_M3:.6f} t/m3")
print("  source: deterministic V_UZ1 tonnes / volume")
print()

print("GRADE DATA")
print(f"  composites        : {len(gc):,}")
print(f"  length-weighted Cu: {COMP_MEAN:.4f} %")
print()


# ======================================================================
# COARSE CANDIDATE BLOCKS
# ======================================================================

tri_pts = triangles.reshape(-1, 3)
vmin = tri_pts.min(axis=0)
vmax = tri_pts.max(axis=0)

bbox = (
    blocks.XC.between(
        vmin[0] - COARSE_PREFILTER_M,
        vmax[0] + COARSE_PREFILTER_M
    )
    & blocks.YC.between(
        vmin[1] - COARSE_PREFILTER_M,
        vmax[1] + COARSE_PREFILTER_M
    )
    & blocks.ZC.between(
        vmin[2] - COARSE_PREFILTER_M,
        vmax[2] + COARSE_PREFILTER_M
    )
)

cand = blocks.loc[bbox].copy()

# Fast approximate distance to triangle centroids only for parent-cell
# prefiltering. Exact distances are calculated after subdivision.
tri_centroids = triangles.mean(axis=1)
tri_tree = cKDTree(tri_centroids)

centre_dist, _ = tri_tree.query(
    cand[["XC", "YC", "ZC"]].to_numpy(float),
    k=1,
    workers=-1,
)

max_tri_radius = float(
    np.max(
        np.linalg.norm(
            triangles - tri_centroids[:, None, :],
            axis=2,
        )
    )
)

# Conservative prefilter: centroid distance minus max triangle radius.
lower_bound_approx = np.maximum(
    centre_dist - max_tri_radius,
    0.0,
)

cand["reference_inside"] = (
    cand.WF.fillna("").astype(str).eq(TARGET_WF)
)

keep_parent = (
    cand.reference_inside
    | (lower_bound_approx <= SHELL_M + 5.0)
)

cand = cand.loc[keep_parent].copy()

# Do not allow material already assigned to a different named wireframe to
# become V_UZ1 during outward perturbation.
wf = cand.WF.fillna("").astype(str).str.strip()

other_named = (
    (~cand.reference_inside)
    & wf.ne("")
    & wf.ne("nan")
    & wf.ne("None")
)

cand["eligible_for_V_UZ1"] = ~other_named

cand["DENSITY"] = pd.to_numeric(
    cand.DENSITY, errors="coerce"
).fillna(density_fallback)

print("COARSE GEOMETRY CANDIDATES")
print(f"  parent cells: {len(cand):,}")
print()


# ======================================================================
# BUILD FINE SUPPORT
# ======================================================================

print("Building fine integration support ...")
support = build_fine_support(cand)

xyz_s = support["xyz"]
vol_s = support["volume"]
den_s = support["density"]
eligible_s = support["eligible"]
parent_ref_s = support["reference_parent"]

N_FINE_SUPPORT_PRE_SCREEN = int(len(xyz_s))
print(f"  fine support points: {N_FINE_SUPPORT_PRE_SCREEN:,}")
print()


# ======================================================================
# EXACT DISTANCE TO TRIANGULATED SURFACE
# ======================================================================

print("Calculating exact point-to-triangle distances ...")
surface_dist, closest_surface = exact_surface_distance(
    xyz_s,
    triangles,
)

# Only retain fine support actually relevant to the stochastic shell,
# plus all support belonging to deterministic reference parents.
keep_support = (
    parent_ref_s
    | (surface_dist <= SHELL_M)
)

xyz_s = xyz_s[keep_support]
vol_s = vol_s[keep_support]
den_s = den_s[keep_support]
eligible_s = eligible_s[keep_support]
parent_ref_s = parent_ref_s[keep_support]
surface_dist = surface_dist[keep_support]
closest_surface = closest_surface[keep_support]

N_FINE_SUPPORT_RETAINED = int(len(xyz_s))
print(f"  retained support points: {N_FINE_SUPPORT_RETAINED:,}")
print()


# ======================================================================
# SIGNED DISTANCE AND REFERENCE RECONCILIATION
# ======================================================================

# The sign is inherited from the deterministic parent-cell membership.
# Because V_UZ1 itself is already finely subblocked, this preserves the
# operational reference interpretation while exact triangle distance controls
# how far each fine support point lies from the surface.
signed_distance = np.where(
    parent_ref_s,
    surface_dist,
    -surface_dist,
)

support_tonnes = vol_s * den_s

reference_support = parent_ref_s.copy()

SUPPORT_REF_VOLUME = float(
    vol_s[reference_support].sum()
)
SUPPORT_REF_TONNES = float(
    support_tonnes[reference_support].sum()
)

rel_tonnage_error = (
    SUPPORT_REF_TONNES - REF_TONNES
) / REF_TONNES

print("REFERENCE SUPPORT RECONCILIATION")
print(f"  block-model volume : {REF_VOLUME:,.3f} m3")
print(f"  support volume     : {SUPPORT_REF_VOLUME:,.3f} m3")
print(f"  block-model tonnes : {REF_TONNES:,.3f} t")
print(f"  support tonnes     : {SUPPORT_REF_TONNES:,.3f} t")
print(f"  relative difference: {100*rel_tonnage_error:.4f} %")
print()

if abs(rel_tonnage_error) > REFERENCE_TONNAGE_REL_TOL:
    raise RuntimeError(
        "Fine-support reference reconciliation failed. "
        f"Tonnage difference = {100*rel_tonnage_error:.3f}%."
    )


# ======================================================================
# GRADE MODEL
# ======================================================================

xyz_c = gc[["X", "Y", "Z"]].to_numpy(float)

grade_centre, eigvals, axes = pca_axes(xyz_c)
pc = (xyz_c - grade_centre) @ axes
pc_extent = np.ptp(pc, axis=0)

logcu = np.log(gc.CU_PCT.to_numpy(float))
logcu_mean = float(
    np.average(logcu, weights=gc.LENGTH.to_numpy(float))
)
resid = logcu - logcu_mean
logvar = float(np.var(resid, ddof=1))

variograms = []
fits = []

for a in range(3):
    vg = directional_variogram(
        pc,
        logcu,
        axis=a,
        seed=SEED,
    )
    vg["direction"] = a + 1
    variograms.append(vg)

    max_range = max(
        RANGE_MIN_M[a] + 0.1,
        RANGE_MAX_FRACTION_OF_EXTENT[a] * pc_extent[a],
    )

    fit = constrained_exp_fit(
        vg,
        variance=logvar,
        min_range=RANGE_MIN_M[a],
        max_range=max_range,
    )
    fits.append(fit)

vg_all = pd.concat(variograms, ignore_index=True)
vg_all.to_csv(
    OUT / "directional_logcu_variograms.csv",
    index=False,
)

ranges = np.array(
    [f["range_parameter"] for f in fits],
    dtype=float,
)

# Preserve broad-to-narrow ordering implied by PCA geometry.
ranges[0] = max(ranges[0], ranges[1])
ranges[2] = min(ranges[2], ranges[1])
ranges = np.maximum(ranges, RANGE_MIN_M)

# Primary mass for every support that becomes mineralised.
support_tonnes_domain_density = (
    vol_s.astype(np.float64, copy=False) * DOMAIN_DENSITY_T_M3
)

print("GRADE SPATIAL MODEL")
print("  PCA extents (m):", np.round(pc_extent, 3))
print("  ranges used (m):", np.round(ranges, 3))
for i, f in enumerate(fits, 1):
    print(
        f"  direction {i}: {f['fit_status']}, "
        f"range={f['range_parameter']:.3f} m"
    )
print()


# ======================================================================
# CONDITIONAL GRADE RANDOM FEATURES
# ======================================================================

rng_master = np.random.default_rng(SEED)

Xc = ((xyz_c - grade_centre) @ axes) / ranges
Xs = ((xyz_s - grade_centre) @ axes) / ranges

omega_grade, phase_grade = make_exponential_rff(
    N_GRADE_FEATURES,
    3,
    rng_master,
)

Phi_c = rff_matrix(
    Xc,
    omega_grade,
    phase_grade,
)

latent_sd = max(0.10, float(np.std(resid, ddof=1)))
Phi_c *= latent_sd

w_mean, w_L = posterior_weights(
    Phi_c,
    resid,
    LOGCU_NOISE_SD,
)

# Grade RFFs are evaluated only on active support when required.

grade_low = float(
    gc.CU_PCT.quantile(GRADE_CAP_LOW_Q)
)
grade_high = float(
    gc.CU_PCT.quantile(GRADE_CAP_HIGH_Q)
)


# ======================================================================
# FIXED-VOLUME, FIXED-DENSITY SIGNED-DISTANCE MODEL
# ======================================================================
# Primary experiment: uncertain boundary position conditional on fixed
# mineralised VOLUME. The authoritative operational support supplies the sign
# of the exact point-to-triangle distance. No HW/FW reconstruction is used.

GEO_FIELD_CHUNK = 50000
GEO_CACHE = RUN_CACHE / "geological_standard_fields.float32.dat"
GEO_CACHE_DONE = RUN_CACHE / "geological_standard_fields.complete"

_tri = mesh_vertices[mesh_faces]
_face_area = 0.5*np.linalg.norm(
    np.cross(_tri[:,1]-_tri[:,0], _tri[:,2]-_tri[:,0]), axis=1)
_vertex_area = np.zeros(len(mesh_vertices), dtype=float)
for _k in range(3):
    np.add.at(_vertex_area, mesh_faces[:,_k], _face_area/3.0)
_vertex_area /= _vertex_area.sum()

_geo_origin = np.average(mesh_vertices, axis=0, weights=_vertex_area)
_Xv = (mesh_vertices-_geo_origin)/BOUNDARY_RANGE_M
_rng = np.random.default_rng(SEED+50000)
omega_geo, phase_geo = make_exponential_rff(N_GEO_FEATURES,3,_rng)
Phi_v = rff_matrix(_Xv,omega_geo,phase_geo).astype(np.float32)

_rngw = np.random.default_rng(SEED + 51000)
# Independent outer geological draws. This matches the assumptions used in
# Proposition 2 and in the finite-inner-simulation correction.
W_geo = _rngw.normal(size=(N_GEO_FEATURES, N_GEO))
_araw = Phi_v.astype(float) @ W_geo
_amean=np.sum(_araw*_vertex_area[:,None],axis=0)
_actr=_araw-_amean[None,:]
_asd=np.sqrt(np.maximum(np.sum(_actr**2*_vertex_area[:,None],axis=0),1e-12))
_astd=_actr/_asd[None,:]

_shape=(len(xyz_s),N_GEO)
_nbytes=_shape[0]*_shape[1]*np.dtype(np.float32).itemsize
if GEO_CACHE.exists() and GEO_CACHE.stat().st_size != _nbytes:
    GEO_CACHE.unlink(missing_ok=True); GEO_CACHE_DONE.unlink(missing_ok=True)
geo_field=np.memmap(GEO_CACHE,dtype=np.float32,
                    mode="r+" if GEO_CACHE.exists() else "w+",shape=_shape)

if not GEO_CACHE_DONE.exists():
    print("Building reusable geological field cache ...")
    for i0 in range(0,len(xyz_s),GEO_FIELD_CHUNK):
        i1=min(i0+GEO_FIELD_CHUNK,len(xyz_s))
        Xb=(closest_surface[i0:i1]-_geo_origin)/BOUNDARY_RANGE_M
        P=rff_matrix(Xb,omega_geo,phase_geo).astype(np.float32)
        raw=P.astype(float)@W_geo
        raw=(raw-_amean[None,:])/_asd[None,:]
        geo_field[i0:i1,:]=raw.astype(np.float32)
        if i1==len(xyz_s) or (i1//GEO_FIELD_CHUNK)%5==0:
            print(f"  geological field cache: {i1:,}/{len(xyz_s):,}")
    geo_field.flush()
    GEO_CACHE_DONE.write_text("complete\n",encoding="utf-8")
else:
    print("Reusing geological field cache.")

def volume_constrained_membership(score, volumes, target, eligible):
    """Select highest-score eligible supports to match target volume.

    The eligibility mask prevents material already assigned to another named
    wireframe from entering V_UZ1. The target volume must be attainable from
    eligible support.
    """
    q = np.asarray(score, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)

    if q.shape != v.shape or q.shape != eligible.shape:
        raise ValueError("score, volumes and eligible must have identical shape")

    candidates = np.flatnonzero(eligible)
    if len(candidates) == 0:
        raise RuntimeError("No eligible support is available for V_UZ1.")

    eligible_volume = float(v[candidates].sum())
    if eligible_volume + 1e-9 < target:
        raise RuntimeError(
            "Target volume exceeds the total V_UZ1-eligible support volume."
        )

    ranked = candidates[np.argsort(q[candidates])[::-1]]
    cv = np.cumsum(v[ranked])
    j = int(np.searchsorted(cv, target, side="left"))
    j = min(max(j, 0), len(ranked) - 1)
    if j > 0 and abs(cv[j - 1] - target) < abs(cv[j] - target):
        j -= 1

    member = np.zeros(len(q), dtype=bool)
    member[ranked[:j + 1]] = True
    threshold = float(q[ranked[j]])

    if np.any(member & ~eligible):
        raise RuntimeError("Ineligible support entered V_UZ1 membership.")

    return member, -threshold, float(cv[j]), float(cv[j] - target)

_v0, _b0, _vv0, _ve0 = volume_constrained_membership(
    signed_distance, vol_s, REF_VOLUME, eligible_s
)

print("FIXED-VOLUME, FIXED-DENSITY SIGNED-DISTANCE MODEL")
print("  reference membership: operational V_UZ1 support")
print("  experiment: boundary-position uncertainty at fixed volume and domain density")
print(f"  target volume: {REF_VOLUME:,.3f} m3")
print(f"  zero-field constrained volume: {_vv0:,.3f} m3")
print(f"  zero-field volume error: {_ve0:+.6f} m3")
print(f"  correlation-range parameter: {BOUNDARY_RANGE_M:g} m")
print(f"  independent geological realisations: {N_GEO}")
print()
# ======================================================================
# FIXED-GEOLOGY GRADE SIMULATION
# ======================================================================

idx_ref = np.flatnonzero(reference_support)
tw_ref = support_tonnes[idx_ref]

rng_fixed = np.random.default_rng(SEED + 1000)

W = draw_weights(
    w_mean,
    w_L,
    N_GRADE,
    rng_fixed,
)

logz = (
    logcu_mean
    + (latent_sd * rff_matrix(Xs[idx_ref], omega_grade, phase_grade)) @ W
)

z = np.exp(logz)
z = np.clip(z, grade_low, grade_high)

fixed_tonnes = np.empty(N_GRADE)
fixed_grade = np.empty(N_GRADE)
fixed_metal = np.empty(N_GRADE)

for r in range(N_GRADE):
    zz = z[:, r]

    if CUTOFF_CU_PCT > 0:
        use = zz >= CUTOFF_CU_PCT
    else:
        use = np.ones(len(zz), dtype=bool)

    tt = tw_ref[use]
    gg = zz[use]

    T = float(tt.sum())
    M = float(np.sum(tt * gg / 100.0))
    A = 100.0 * M / T if T > 0 else np.nan

    fixed_tonnes[r] = T
    fixed_grade[r] = A
    fixed_metal[r] = M


# ======================================================================
# NESTED SIMULATION
# ======================================================================
summary_rows=[]
geo_rows=[]

for scenario_i,boundary_sd in enumerate(BOUNDARY_SDS_M):
    print(f"Running nested scenario: boundary SD = {boundary_sd:g} m")
    tonnes_mat=np.empty((N_GEO,N_GRADE))
    grade_mat=np.empty((N_GEO,N_GRADE))
    metal_mat=np.empty((N_GEO,N_GRADE))

    checkpoint_npz=RUN_CACHE/f"checkpoint_volume_constrained_boundary_sd_{boundary_sd:g}m.npz"
    checkpoint_geo=RUN_CACHE/f"checkpoint_volume_constrained_geometry_boundary_sd_{boundary_sd:g}m.csv"
    start_g=0
    if checkpoint_npz.exists():
        if not checkpoint_geo.exists():
            raise RuntimeError(f"Checkpoint exists without geometry table: {checkpoint_geo}")
        with np.load(checkpoint_npz, allow_pickle=False) as cp:
            cp_signature = str(cp["run_signature"]) if "run_signature" in cp else None
            if cp_signature != RUN_SIGNATURE:
                raise RuntimeError(f"Checkpoint run signature mismatch: {checkpoint_npz}")
            if not (cp["tonnes"].shape == tonnes_mat.shape and
                    cp["grade"].shape == grade_mat.shape and
                    cp["metal"].shape == metal_mat.shape):
                raise RuntimeError(f"Checkpoint dimensions do not match: {checkpoint_npz}")
            start_g = int(cp["completed"])
            if not (0 <= start_g <= N_GEO):
                raise RuntimeError(f"Invalid checkpoint progress: {start_g}")
            tonnes_mat[:] = cp["tonnes"]
            grade_mat[:] = cp["grade"]
            metal_mat[:] = cp["metal"]
        restored_geo = pd.read_csv(checkpoint_geo)
        if len(restored_geo) < start_g:
            raise RuntimeError(
                f"Checkpoint geometry has only {len(restored_geo)} records "
                f"for {start_g} completed draws at boundary SD {boundary_sd}."
            )
        if len(restored_geo) > start_g:
            # A crash between writing geometry and committing the NPZ can
            # leave a newer CSV than the last fully committed simulation.
            # The NPZ's completed counter is authoritative.
            print(f"  Discarding {len(restored_geo) - start_g} "
                  "uncommitted geometry checkpoint rows.")
            restored_geo = restored_geo.iloc[:start_g].copy()
        geo_rows.extend(restored_geo.to_dict("records"))
        print(f"  Resuming from geological realisation {start_g+1}/{N_GEO}")

    for g in range(start_g,N_GEO):
        print(f"  geology {g+1:02d}/{N_GEO}: solving fixed-volume position ...")
        score=(signed_distance.astype(np.float64,copy=False)+
               boundary_sd*np.asarray(geo_field[:,g],dtype=np.float64))
        member, b_g, realised_volume, volume_error = volume_constrained_membership(
            score, vol_s, REF_VOLUME, eligible_s
        )
        idx=np.flatnonzero(member)
        geological_tonnes=float(support_tonnes_domain_density[idx].sum())

        geo_rows.append({
            "boundary_sd_m":boundary_sd,
            "geology_realisation":g+1,
            "n_support_points":len(idx),
            "geological_volume_m3":realised_volume,
            "volume_error_m3":volume_error,
            "volume_ratio_to_reference":realised_volume/REF_VOLUME,
            "geological_tonnes_before_cutoff":geological_tonnes,
            "tonnage_ratio_to_reference":geological_tonnes/REF_TONNES,
            "inherited_density_tonnes_sensitivity":float(support_tonnes[idx].sum()),
            "inherited_density_tonnage_ratio":float(support_tonnes[idx].sum())/REF_TONNES,
            "domain_density_t_m3":DOMAIN_DENSITY_T_M3,
            "centering_offset_m":b_g,
            "surface_field_sd_m":float(boundary_sd),
            "maximum_anchor_displacement_m":
                float(boundary_sd * np.max(np.abs(_astd[:, g]))),
            "selected_ineligible_support_count": int(np.sum(member & ~eligible_s)),
            "geology_method": "fixed_volume_fixed_density_signed_distance",
        })

        rng_grade=np.random.default_rng(
            SEED+100000*(scenario_i+1)+1000*(g+1))
        Wg=draw_weights(w_mean,w_L,N_GRADE,rng_grade)
        logzg=(logcu_mean+
               (latent_sd*rff_matrix(Xs[idx],omega_grade,phase_grade))@Wg)
        zg=np.clip(np.exp(logzg),grade_low,grade_high)
        tw=support_tonnes_domain_density[idx]
        if CUTOFF_CU_PCT>0:
            use=zg>=CUTOFF_CU_PCT
            tmat=tw[:,None]*use
            Tvec=tmat.sum(axis=0)
            Mvec=(tmat*zg/100.0).sum(axis=0)
        else:
            Tvec=np.full(N_GRADE,tw.sum(),dtype=float)
            Mvec=(tw[:,None]*zg/100.0).sum(axis=0)
        Avec=np.divide(100.0*Mvec,Tvec,out=np.full_like(Mvec,np.nan),
                       where=Tvec>0)
        tonnes_mat[g,:]=Tvec; grade_mat[g,:]=Avec; metal_mat[g,:]=Mvec

        # Commit geometry first and the simulation NPZ second. On restart,
        # the committed NPZ counter is authoritative. Any newer uncommitted
        # geometry rows can be trimmed without changing numerical results.
        geometry_tmp = checkpoint_geo.with_name(checkpoint_geo.name + ".tmp")
        pd.DataFrame([r for r in geo_rows if r["boundary_sd_m"] == boundary_sd]
                    ).to_csv(geometry_tmp, index=False)
        geometry_tmp.replace(checkpoint_geo)
        checkpoint_tmp = checkpoint_npz.with_name(checkpoint_npz.stem + ".tmp.npz")
        np.savez_compressed(checkpoint_tmp, tonnes=tonnes_mat,
                            grade=grade_mat, metal=metal_mat, completed=g+1,
                            run_signature=RUN_SIGNATURE)
        checkpoint_tmp.replace(checkpoint_npz)

    for metric_name,mat,ref_value,fixed_values in (
        ("tonnes",tonnes_mat,REF_TONNES,fixed_tonnes),
        ("average_grade",grade_mat,REF_GRADE,fixed_grade),
        ("contained_metal",metal_mat,REF_METAL,fixed_metal),
    ):
        bar_g=np.nanmean(mat,axis=1)
        s2_g=np.nanvar(mat,axis=1,ddof=1)
        U_grade=float(np.nanmean(s2_g))
        S_bar=float(np.nanvar(bar_g,ddof=1))
        U_geo_raw=float(S_bar-U_grade/N_GRADE)
        U_geo_report=max(0.0,U_geo_raw)
        den=U_grade+U_geo_report
        R_G=U_geo_report/den if den>0 else np.nan
        full_var=float(np.nanvar(mat.reshape(-1),ddof=1))
        fixed_var=float(np.nanvar(fixed_values,ddof=1))
        flat=mat.reshape(-1)
        summary_rows.append({
            "boundary_sd_m":boundary_sd,"metric":metric_name,
            "reference_value":ref_value,"mean":float(np.nanmean(flat)),
            "p05":float(np.nanquantile(flat,.05)),
            "p50":float(np.nanquantile(flat,.50)),
            "p95":float(np.nanquantile(flat,.95)),
            "U_grade":U_grade,"U_geo_raw":U_geo_raw,
            "U_geo_report":U_geo_report,"R_G":R_G,
            "fixed_to_full_variance_ratio":
                fixed_var/full_var if full_var>0 else np.nan,
        })

# Publish the final per-scenario geometry tables at their historical paths,
# while resumption uses only the fingerprint-isolated checkpoint copies.
for _sd in BOUNDARY_SDS_M:
    _gsrc = RUN_CACHE / f"checkpoint_volume_constrained_geometry_boundary_sd_{_sd:g}m.csv"
    if _gsrc.exists():
        shutil.copy2(_gsrc, OUT / _gsrc.name)

summary = pd.DataFrame(summary_rows)
geo_summary = pd.DataFrame(geo_rows)

if int(geo_summary["selected_ineligible_support_count"].sum()) != 0:
    raise RuntimeError("At least one ineligible support point entered V_UZ1.")

# Hard validation of the primary constraint.
_max_abs_volume_error = float(np.max(np.abs(
    geo_summary["geological_volume_m3"].to_numpy(float) - REF_VOLUME)))
_max_abs_tonnage_rel_error = float(np.max(np.abs(
    geo_summary["geological_tonnes_before_cutoff"].to_numpy(float)
    / REF_TONNES - 1.0)))
print("PRIMARY CONSTRAINT VALIDATION")
print(f"  maximum absolute volume error: {_max_abs_volume_error:.6f} m3")
print(f"  maximum absolute primary tonnage relative error: {100*_max_abs_tonnage_rel_error:.6f} %")
if _max_abs_tonnage_rel_error > 1e-3:
    raise RuntimeError(
        "Primary tonnage failed to reconcile within 0.1% of reference.")
print("  status: PASS")
print()


expected_summary_rows = len(BOUNDARY_SDS_M) * 3
if len(summary) != expected_summary_rows:
    raise RuntimeError(
        f"Expected {expected_summary_rows} scenario-summary rows, "
        f"but obtained {len(summary)}."
    )

summary.to_csv(
    OUT / "real_case_scenario_summary.csv",
    index=False,
)

geo_summary.to_csv(
    OUT / "geological_realisation_summary.csv",
    index=False,
)

reference_reconciliation = pd.DataFrame([{
    "reference_blocks": len(ref),
    "block_model_volume_m3": REF_VOLUME,
    "fine_support_reference_volume_m3": SUPPORT_REF_VOLUME,
    "block_model_tonnes": REF_TONNES,
    "fine_support_reference_tonnes": SUPPORT_REF_TONNES,
    "relative_tonnage_difference": rel_tonnage_error,
    "block_model_Cu_pct": REF_GRADE,
    "block_model_contained_Cu_t": REF_METAL,
    "composite_count": len(gc),
    "composite_length_weighted_Cu_pct": COMP_MEAN,
}])

reference_reconciliation.to_csv(
    OUT / "reference_reconciliation.csv",
    index=False,
)

anisotropy = pd.DataFrame({
    "direction": [1, 2, 3],
    "PCA_extent_m": pc_extent,
    "range_used_m": ranges,
    "nugget": [f["nugget"] for f in fits],
    "partial_sill": [f["partial_sill"] for f in fits],
    "sill": [f["sill"] for f in fits],
    "fit_status": [f["fit_status"] for f in fits],
})

anisotropy.to_csv(
    OUT / "grade_anisotropy.csv",
    index=False,
)


# ======================================================================
# LATEX RESULTS FILE FOR THE MANUSCRIPT
# ======================================================================

def _tex_number(value, decimals, thousands=False):
    text = f"{float(value):,.{decimals}f}" if thousands else f"{float(value):.{decimals}f}"
    return text.replace(",", "{,}")


def _summary_row(boundary_sd, metric):
    q = summary.loc[
        (summary.boundary_sd_m == boundary_sd)
        & (summary.metric == metric)
    ]
    if len(q) != 1:
        raise RuntimeError(
            f"Expected one summary row for boundary SD {boundary_sd} and {metric}."
        )
    return q.iloc[0]


def write_manuscript_results():
    sd1, sd2, sd5 = BOUNDARY_SDS_M
    g1 = _summary_row(sd1, "average_grade")
    g2 = _summary_row(sd2, "average_grade")
    g5 = _summary_row(sd5, "average_grade")
    m1 = _summary_row(sd1, "contained_metal")
    m2 = _summary_row(sd2, "contained_metal")
    m5 = _summary_row(sd5, "contained_metal")

    mean_tonnes = []
    for sd in BOUNDARY_SDS_M:
        q = geo_summary.loc[
            geo_summary.boundary_sd_m == sd,
            "geological_tonnes_before_cutoff",
        ]
        mean_tonnes.append(float(q.mean()))

    r_values = np.array([g1.R_G, g2.R_G, g5.R_G, m1.R_G, m2.R_G, m5.R_G], float)
    r_lo = int(np.floor(100.0 * np.nanmin(r_values)))
    r_hi = int(np.ceil(100.0 * np.nanmax(r_values)))
    if r_lo == r_hi:
        r_text = str(r_lo)
    else:
        r_text = f"{r_lo}--{r_hi}"

    table_rows = [
        f"1 & Average grade (\\% Cu) & {_tex_number(g1['mean'],3)} & {_tex_number(g1.p05,3)} & {_tex_number(g1.p50,3)} & {_tex_number(g1.p95,3)} & {_tex_number(g1.U_grade,4)} & {_tex_number(g1.U_geo_report,4)} & {_tex_number(g1.R_G,4)} \\\\",
        f"1 & Contained metal (t Cu) & {_tex_number(m1['mean'],1)} & {_tex_number(m1.p05,1)} & {_tex_number(m1.p50,1)} & {_tex_number(m1.p95,1)} & {_tex_number(m1.U_grade,0)} & {_tex_number(m1.U_geo_report,0)} & {_tex_number(m1.R_G,4)} \\\\",
        f"2 & Average grade (\\% Cu) & {_tex_number(g2['mean'],3)} & {_tex_number(g2.p05,3)} & {_tex_number(g2.p50,3)} & {_tex_number(g2.p95,3)} & {_tex_number(g2.U_grade,4)} & {_tex_number(g2.U_geo_report,4)} & {_tex_number(g2.R_G,4)} \\\\",
        f"2 & Contained metal (t Cu) & {_tex_number(m2['mean'],1)} & {_tex_number(m2.p05,1)} & {_tex_number(m2.p50,1)} & {_tex_number(m2.p95,1)} & {_tex_number(m2.U_grade,0)} & {_tex_number(m2.U_geo_report,0)} & {_tex_number(m2.R_G,4)} \\\\",
        f"5 & Average grade (\\% Cu) & {_tex_number(g5['mean'],3)} & {_tex_number(g5.p05,3)} & {_tex_number(g5.p50,3)} & {_tex_number(g5.p95,3)} & {_tex_number(g5.U_grade,4)} & {_tex_number(g5.U_geo_report,4)} & {_tex_number(g5.R_G,4)} \\\\",
        f"5 & Contained metal (t Cu) & {_tex_number(m5['mean'],1)} & {_tex_number(m5.p05,1)} & {_tex_number(m5.p50,1)} & {_tex_number(m5.p95,1)} & {_tex_number(m5.U_grade,0)} & {_tex_number(m5.U_geo_report,0)} & {_tex_number(m5.R_G,4)} \\\\",
    ]

    macros = {
        "MontyPCAExtentOne": _tex_number(pc_extent[0], 3),
        "MontyPCAExtentTwo": _tex_number(pc_extent[1], 3),
        "MontyPCAExtentThree": _tex_number(pc_extent[2], 3),
        "MontyRangeOne": _tex_number(ranges[0], 3),
        "MontyRangeTwo": _tex_number(ranges[1], 3),
        "MontyRangeThree": _tex_number(ranges[2], 3),
        "MontyFineSupportBuilt": f"{N_FINE_SUPPORT_PRE_SCREEN:,}".replace(",", "{,}"),
        "MontyFineSupportRetained": f"{N_FINE_SUPPORT_RETAINED:,}".replace(",", "{,}"),
        "MontyMaxVolumeError": _tex_number(_max_abs_volume_error, 6),
        "MontyMaxVolumeErrorPct": _tex_number(100.0 * _max_abs_volume_error / REF_VOLUME, 6),
        "MontyMaxTonnageErrorPct": _tex_number(100.0 * _max_abs_tonnage_rel_error, 6),
        "MontyMeanTonnageOne": _tex_number(mean_tonnes[0], 3, True),
        "MontyMeanTonnageTwo": _tex_number(mean_tonnes[1], 3, True),
        "MontyMeanTonnageFive": _tex_number(mean_tonnes[2], 3, True),
        "MontyRGGradeOne": _tex_number(g1.R_G, 4),
        "MontyRGGradeTwo": _tex_number(g2.R_G, 4),
        "MontyRGGradeFive": _tex_number(g5.R_G, 4),
        "MontyRGMetalOne": _tex_number(m1.R_G, 4),
        "MontyRGMetalTwo": _tex_number(m2.R_G, 4),
        "MontyRGMetalFive": _tex_number(m5.R_G, 4),
        "MontyRGPercentRange": r_text,
        "MontyUGeoGradeOne": _tex_number(g1.U_geo_report, 4),
        "MontyUGeoGradeTwo": _tex_number(g2.U_geo_report, 4),
        "MontyUGeoGradeFive": _tex_number(g5.U_geo_report, 4),
        "MontyUGeoMetalOne": _tex_number(m1.U_geo_report, 0),
        "MontyUGeoMetalTwo": _tex_number(m2.U_geo_report, 0),
        "MontyUGeoMetalFive": _tex_number(m5.U_geo_report, 0),
        "MontyMeanGradeOne": _tex_number(g1['mean'], 3),
        "MontyMeanGradeTwo": _tex_number(g2['mean'], 3),
        "MontyMeanGradeFive": _tex_number(g5['mean'], 3),
        "MontyMeanMetalOne": _tex_number(m1['mean'], 1, True),
        "MontyMeanMetalTwo": _tex_number(m2['mean'], 1, True),
        "MontyMeanMetalFive": _tex_number(m5['mean'], 1, True),
    }

    lines = [
        "% Auto-generated by real_case_geological_grade_uncertainty_FINAL.py",
        "% Do not edit numerical values by hand.",
    ]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    lines.append("\\newcommand{\\MontyTableRows}{%")
    lines.extend(table_rows)
    lines.append("}")
    text = "\n".join(lines) + "\n"

    for path in [ROOT / "real_case_results.tex", OUT / "real_case_results.tex"]:
        path.write_text(text, encoding="utf-8")


write_manuscript_results()


# ======================================================================
# CONVERGENCE
# ======================================================================

for _sd in BOUNDARY_SDS_M:
    _cp = RUN_CACHE / f"checkpoint_volume_constrained_boundary_sd_{_sd:g}m.npz"
    if not _cp.exists():
        raise FileNotFoundError(
            f"Required corrected-analysis checkpoint not found: {_cp}. "
            "Run or resume the corresponding scenario first."
        )

conv_rows = []

middle_sd = BOUNDARY_SDS_M[
    len(BOUNDARY_SDS_M) // 2
]

# Reuse the corrected production checkpoint for the middle boundary-SD
# scenario.  The checkpoint stores the matrices under the short internal
# keys tonnes / grade / metal.
npz = np.load(
    RUN_CACHE / f"checkpoint_volume_constrained_boundary_sd_{middle_sd:g}m.npz"
)

_metric_npz_key = {
    "tonnes": "tonnes",
    "average_grade": "grade",
    "contained_metal": "metal",
}

for metric in [
    "tonnes",
    "average_grade",
    "contained_metal",
]:
    mat = npz[_metric_npz_key[metric]]

    for Gsub in [10, 20, 30, 40, 50]:
        if Gsub > N_GEO:
            continue

        for Rsub in [10, 20, 30, 40, 50]:
            if Rsub > N_GRADE:
                continue

            d = decompose_nested(
                mat[:Gsub, :Rsub]
            )

            conv_rows.append({
                "boundary_sd_m": middle_sd,
                "metric": metric,
                "G": Gsub,
                "R": Rsub,
                "U_grade": d["U_grade"],
                "U_geo_raw": d["U_geo_raw"],
                "U_geo_report": d["U_geo_report"],
                "R_G": d["R_G"],
            })

pd.DataFrame(conv_rows).to_csv(
    OUT / "convergence_summary.csv",
    index=False,
)


# ======================================================================
# FIXED-VOLUME / FIXED-DENSITY DIAGNOSTICS
# ======================================================================

geo_diag_rows = []

for sd in BOUNDARY_SDS_M:
    q = geo_summary[
        geo_summary.boundary_sd_m == sd
    ]

    geo_diag_rows.append({
        "boundary_sd_m": sd,
        "reference_tonnes": REF_TONNES,
        "mean_geological_tonnes": q.geological_tonnes_before_cutoff.mean(),
        "median_geological_tonnes": q.geological_tonnes_before_cutoff.median(),
        "sd_geological_tonnes": q.geological_tonnes_before_cutoff.std(ddof=1),
        "p05_geological_tonnes": q.geological_tonnes_before_cutoff.quantile(.05),
        "p95_geological_tonnes": q.geological_tonnes_before_cutoff.quantile(.95),
        "mean_to_reference_ratio": (
            q.geological_tonnes_before_cutoff.mean() / REF_TONNES
        ),
        "median_to_reference_ratio": (
            q.geological_tonnes_before_cutoff.median() / REF_TONNES
        ),
    })

geo_diag = pd.DataFrame(geo_diag_rows)

geo_diag.to_csv(
    OUT / "geometry_bias_diagnostics.csv",
    index=False,
)

# Hard geometry sanity check. The 1 m case is expected to remain close to
# the deterministic reference under a symmetric local contact perturbation.
# This threshold is deliberately a diagnostic, not a statistical theorem.
q1 = geo_diag.loc[geo_diag.boundary_sd_m == min(BOUNDARY_SDS_M)]
if len(q1):
    ratio1 = float(q1.iloc[0].mean_to_reference_ratio)
    if not (0.90 <= ratio1 <= 1.10):
        print()
        print("WARNING: 1 m primary scenario fails fixed-volume/fixed-density reconciliation.")
        print(f"  mean/reference tonnes = {ratio1:.4f}")
        print("  Do not interpret R_G substantively unless fixed-volume/fixed-density diagnostics are acceptable.")



# ======================================================================
# PUBLICATION FIGURES
# ======================================================================

# The main-paper figures are limited to the plots that
# demonstrate (i) the geological object and conditioning data,
# (ii) the grade-continuity model, and
# (iii) the sensitivity of the geological variance share.
#
# Additional geometry/tonnage diagnostics are retained as supplementary
# figures because they are important validation checks but are not the
# central scientific result.

FIG_MAIN = OUT / "figures_main"
FIG_SUPP = OUT / "figures_supplementary"
FIG_MAIN.mkdir(exist_ok=True)
FIG_SUPP.mkdir(exist_ok=True)


def save_publication_figure(fig, stem, folder, dpi=600):
    """Save NRR/Springer artwork with reliably embedded lettering.

    EPS is the preferred submission format for predominantly vector figures.
    TIFF at 600 dpi is available for combination or colour artwork. PDF is
    retained for reliable local pdfLaTeX compilation and PNG for inspection.
    """
    fig.savefig(
        folder / f"{stem}.eps",
        format="eps",
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
    )
    fig.savefig(
        folder / f"{stem}.pdf",
        format="pdf",
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
    )
    fig.savefig(
        folder / f"{stem}.tiff",
        format="tiff",
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        folder / f"{stem}.png",
        format="png",
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
    )

    # Main numbered figures are also copied beside the manuscript. The PDF
    # companion is used by pdfLaTeX; EPS or TIFF can be uploaded as artwork.
    if folder == FIG_MAIN and stem.startswith("Fig") and stem[3:].isdigit():
        for ext in ("eps", "pdf", "tiff", "png"):
            src = folder / f"{stem}.{ext}"
            shutil.copy2(src, ROOT / src.name)


# ----------------------------------------------------------------------
# Fig. 9. Reference V_UZ1 geometry and conditioning composites
# ----------------------------------------------------------------------
# Plot the actual triangulated wireframe.  To keep the vector PDF usable,
# a regular subset of faces is displayed when the mesh is very dense.
# This affects plotting only; the analysis always uses the full mesh.

max_plot_faces = 6000
if len(triangles) > max_plot_faces:
    face_step = int(np.ceil(len(triangles) / max_plot_faces))
    plot_triangles = triangles[::face_step]
else:
    plot_triangles = triangles

fig = plt.figure(figsize=(6.85, 5.6))
ax = fig.add_subplot(111, projection="3d")

mesh = Poly3DCollection(
    plot_triangles,
    alpha=0.18,
    linewidths=0.15,
    edgecolors="0.35",
    facecolors="0.75",
)
ax.add_collection3d(mesh)

sc = ax.scatter(
    gc.X.to_numpy(float),
    gc.Y.to_numpy(float),
    gc.Z.to_numpy(float),
    c=gc.CU_PCT.to_numpy(float),
    cmap="viridis",
    s=24,
    depthshade=False,
)

# Force axes to include the complete wireframe and composites.
all_x = np.r_[tri_pts[:, 0], gc.X.to_numpy(float)]
all_y = np.r_[tri_pts[:, 1], gc.Y.to_numpy(float)]
all_z = np.r_[tri_pts[:, 2], gc.Z.to_numpy(float)]

ax.set_xlim(all_x.min(), all_x.max())
ax.set_ylim(all_y.min(), all_y.max())
ax.set_zlim(all_z.min(), all_z.max())

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

cbar = fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.08)
cbar.set_label("Cu (%)")

# A reproducible viewing angle chosen to show both vertical and lateral
# geometry. Change these two values only if a different orientation is
# preferable after visual inspection.
ax.view_init(elev=22, azim=-62)

# Use approximately equal physical scaling in the three dimensions.
try:
    ax.set_box_aspect((
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z),
    ))
except Exception:
    pass

fig.tight_layout()
save_publication_figure(
    fig,
    "Fig9",
    FIG_MAIN,
)
plt.close(fig)


# ----------------------------------------------------------------------
# Fig. 10. Directional log-Cu variograms and fitted models
# ----------------------------------------------------------------------
# Separate panels prevent the poorly identified long direction from
# visually compressing the shorter-range directions.

fig, axs = plt.subplots(
    1, 3,
    figsize=(11.0, 3.8),
    sharey=True,
)

for direction, ax in zip([1, 2, 3], axs):
    q = vg_all.loc[vg_all.direction == direction].copy()
    fit = fits[direction - 1]

    if len(q):
        ax.scatter(
            q.lag_m,
            q.semivariance,
            s=26,
            label="Empirical",
        )

        xmax = float(q.lag_m.max())
        hh = np.linspace(0.0, xmax, 250)

        gg = (
            fit["nugget"]
            + fit["partial_sill"]
            * (1.0 - np.exp(-hh / fit["range_parameter"]))
        )

        ax.plot(
            hh,
            gg,
            linewidth=1.5,
            label="Fitted model",
        )
    ax.set_xlabel("Lag (m)")
    ax.grid(alpha=0.2)

    status_labels = {
        "range_bound_active": "Range at upper bound",
        "constrained_fit": "Constrained fit",
        "fallback_sparse_direction": "Sparse-data fallback",
    }
    status = status_labels.get(str(fit.get("fit_status", "")), "Fit status unknown")
    range_used = float(ranges[direction - 1])

    ax.text(
        0.04,
        0.96,
        f"Range used = {range_used:.1f} m\n{status}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )

axs[0].set_ylabel("Semivariance of log(Cu)")
axs[0].legend(loc="best", fontsize=8)
fig.tight_layout()
save_publication_figure(
    fig,
    "Fig10",
    FIG_MAIN,
)
plt.close(fig)


# ----------------------------------------------------------------------
# Fig. 11. Geological tonnage distributions
# ----------------------------------------------------------------------
# This figure is a numerical validation diagnostic for the fixed-volume,
# fixed-density primary experiment. The x-axis is formatted in actual
# tonnes rather than Matplotlib offset notation so that the very small
# residual variation is immediately interpretable.

from matplotlib.ticker import StrMethodFormatter

fig, axs = plt.subplots(
    1,
    len(BOUNDARY_SDS_M),
    figsize=(11.0, 3.8),
    sharex=True,
    sharey=True,
)

if len(BOUNDARY_SDS_M) == 1:
    axs = [axs]

all_geo_tonnes = geo_summary["geological_tonnes_before_cutoff"].to_numpy(float)
common_min = min(float(np.nanmin(all_geo_tonnes)), REF_TONNES)
common_max = max(float(np.nanmax(all_geo_tonnes)), REF_TONNES)
common_bins = np.linspace(common_min, common_max, 16)

for ax, sd in zip(axs, BOUNDARY_SDS_M):
    q = geo_summary.loc[
        geo_summary.boundary_sd_m == sd,
        "geological_tonnes_before_cutoff",
    ].to_numpy(float)

    ax.hist(q, bins=common_bins, alpha=0.75)
    ax.axvline(
        REF_TONNES,
        linestyle="--",
        linewidth=1.4,
        label="Reference",
    )
    ax.axvline(
        np.mean(q),
        linestyle="-",
        linewidth=1.2,
        label="Mean",
    )
    ax.set_xlabel("Tonnes")
    # Display actual tonnage values (e.g. 24,535) rather than +2.453e4
    # offset notation. Whole tonnes are sufficient for this diagnostic.
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.15)

axs[0].set_ylabel("Frequency")
axs[-1].legend(fontsize=8)
fig.tight_layout()
save_publication_figure(
    fig,
    "Fig11",
    FIG_MAIN,
)
plt.close(fig)


# ----------------------------------------------------------------------
# Fig. 12. Geological variance share versus boundary uncertainty
# ----------------------------------------------------------------------
# Tonnes are intentionally omitted. With zero Cu cut-off, conditional
# grade simulation cannot change tonnes, so R_G = 1 for tonnes by
# construction whenever geological variance is positive. Plotting that
# value as though it were an empirical finding would be misleading.
# Use a labelled expanded vertical scale so differences between high
# geological variance shares are legible. Interpret the differences alongside
# the absolute variance components; the axis no longer starts at zero.

rg_plot = summary.loc[
    summary["metric"].isin(["average_grade", "contained_metal"])
].copy()

metric_labels = {
    "average_grade": "Average Cu grade",
    "contained_metal": "Contained Cu",
}

fig, ax = plt.subplots(figsize=(6.85, 4.6))

metric_styles = {
    "average_grade": {"marker": "o", "linestyle": "-"},
    "contained_metal": {"marker": "s", "linestyle": "--"},
}

for metric in ["average_grade", "contained_metal"]:
    q = rg_plot.loc[rg_plot.metric == metric].sort_values("boundary_sd_m")
    if len(q):
        ax.plot(
            q.boundary_sd_m,
            q.R_G,
            linewidth=1.6,
            label=metric_labels[metric],
            **metric_styles[metric],
        )

ax.set_xlabel("Boundary uncertainty SD (m)")
ax.set_ylabel(r"Geological variance share $R_G$")
ax.set_xticks(list(BOUNDARY_SDS_M))

# Optimise the vertical axis from the plotted results rather than forcing
# 0--1. For the current Monty shares this produces approximately 0.970--0.985.
# Round outward to simple tick boundaries and leave a small visible margin.
rg_values = rg_plot["R_G"].to_numpy(dtype=float)
rg_values = rg_values[np.isfinite(rg_values)]
if not len(rg_values):
    raise ValueError("Fig12 has no finite geological variance shares to plot")

span = float(np.ptp(rg_values))
tick_step = (0.005 if span <= 0.03 else
             0.01 if span <= 0.06 else
             0.05 if span <= 0.30 else 0.20)
pad = max(0.002, 0.12 * span)
y_lower = max(0.0, tick_step * np.floor((rg_values.min() - pad) / tick_step))
y_upper = min(1.0, tick_step * np.ceil((rg_values.max() + pad) / tick_step))
if y_upper - y_lower < 2.0 * tick_step:
    y_lower = max(0.0, y_lower - tick_step)
    y_upper = min(1.0, y_upper + tick_step)

ax.set_ylim(y_lower, y_upper)
ax.set_yticks(np.arange(y_lower, y_upper + 0.5 * tick_step, tick_step))
ax.yaxis.set_major_formatter("{x:.3f}" if tick_step < 0.01 else "{x:.2f}")
ax.grid(alpha=0.2)
ax.legend()
ax.text(0.98, 0.03, "Expanded vertical scale",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="0.4")
fig.tight_layout()
save_publication_figure(
    fig,
    "Fig12",
    FIG_MAIN,
)
plt.close(fig)


# ======================================================================
# SUPPLEMENTARY DIAGNOSTIC FIGURES
# ======================================================================

# S1. Mean and median tonnage relative to the deterministic reference.
fig, ax = plt.subplots(figsize=(6.85, 4.6))

ax.plot(
    geo_diag.boundary_sd_m,
    geo_diag.mean_to_reference_ratio,
    marker="o",
    linestyle="-",
    linewidth=1.5,
    label="Mean/reference",
)

ax.plot(
    geo_diag.boundary_sd_m,
    geo_diag.median_to_reference_ratio,
    marker="s",
    linestyle="--",
    linewidth=1.5,
    label="Median/reference",
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.4,
    label="Deterministic reference",
)

ax.set_xlabel("Boundary uncertainty SD (m)")
ax.set_ylabel("Geological tonnes / reference tonnes")
ax.set_xticks(list(BOUNDARY_SDS_M))
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
save_publication_figure(
    fig,
    "FigS2",
    FIG_SUPP,
)
plt.close(fig)


# S2. Monte Carlo convergence of R_G for the middle boundary scenario.
conv_df = pd.read_csv(OUT / "convergence_summary.csv")

fig, axs = plt.subplots(
    1, 2,
    figsize=(9.0, 3.8),
    sharey=True,
)

for ax, metric in zip(
    axs,
    ["average_grade", "contained_metal"],
):
    q = conv_df.loc[conv_df.metric == metric].copy()

    convergence_styles = [("o", "-"), ("s", "--"), ("^", ":"), ("D", "-."), ("v", "-")]
    for style_i, rsub in enumerate(sorted(q.R.unique())):
        qq = q.loc[q.R == rsub].sort_values("G")
        marker, linestyle = convergence_styles[style_i % len(convergence_styles)]
        ax.plot(
            qq.G,
            qq.R_G,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.1,
            label=f"R = {int(rsub)}",
        )
    ax.set_xlabel("Number of geological realisations, G")
    ax.grid(alpha=0.2)

axs[0].set_ylabel(r"Geological variance share $R_G$")
axs[0].set_ylim(0.0, 1.0)
axs[-1].legend(
    title="Grade realisations",
    fontsize=7,
    title_fontsize=7,
)
fig.tight_layout()
save_publication_figure(
    fig,
    "Fig_S2_RG_convergence",
    FIG_SUPP,
)
plt.close(fig)

# ======================================================================
# METADATA
# ======================================================================

metadata = {
    "version": "certified-distance fixed-volume signed-distance analysis",
    "run_signature": RUN_SIGNATURE,
    "run_provenance": RUN_PROVENANCE,
    "exact_distance_search": "KD-tree broadphase with certified global-radius fallback",
    "seed": SEED,
    "block_file": BLOCK_FILE.name,
    "composite_file": COMP_FILE.name,
    "wireframe_file": DXF_FILE.name,
    "target_wireframe": TARGET_WF,
    "target_composite_domain": TARGET_COMP_DOMAIN,
    "n_blocks_full": int(len(blocks)),
    "n_reference_blocks": int(len(ref)),
    "n_grade_composites": int(len(gc)),
    "n_dxf_triangles": int(len(triangles)),
    "n_fine_support_points_before_screening": N_FINE_SUPPORT_PRE_SCREEN,
    "n_fine_support_points_retained": N_FINE_SUPPORT_RETAINED,
    "reference_volume_m3": REF_VOLUME,
    "reference_tonnes": REF_TONNES,
    "reference_Cu_pct": REF_GRADE,
    "reference_contained_Cu_t": REF_METAL,
    "fine_support_reference_volume_m3": SUPPORT_REF_VOLUME,
    "fine_support_reference_tonnes": SUPPORT_REF_TONNES,
    "fine_support_relative_tonnage_error": rel_tonnage_error,
    "fine_support_target_m": FINE_SUPPORT_M,
    "shell_m": SHELL_M,
    "boundary_sds_m": list(BOUNDARY_SDS_M),
    "boundary_range_m": BOUNDARY_RANGE_M,
    "geology_method": "fixed-volume fixed-density signed-distance boundary-position perturbation",
    "independent_geological_realisations": True,
    "eligibility_mask_enforced": True,
    "rff_covariance_kernel": "isotropic exponential after anisotropic coordinate scaling",
    "N_GEO": N_GEO,
    "N_GRADE": N_GRADE,
    "cutoff_Cu_pct": CUTOFF_CU_PCT,
    "grade_PCA_eigenvalues": eigvals.tolist(),
    "grade_PCA_axes_columns": axes.tolist(),
    "grade_ranges_m_used": ranges.tolist(),
    "grade_variogram_fits": fits,
    "methodological_note": (
        "Operational geology and grade data are real. Boundary uncertainty "
        "magnitude and boundary correlation range are sensitivity parameters, "
        "not empirically calibrated deposit uncertainty."
    ),
    "geometry_note": (
        "The final implementation uses exact point-to-triangle distance, "
        "closest-surface random fields, independent geological draws, fine "
        "integration support, fixed volume, fixed domain density and an "
        "explicit eligibility mask that excludes other named wireframes."
    ),
}

with open(
    OUT / "real_case_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)


# ======================================================================
# FINAL CONSOLE OUTPUT
# ======================================================================

print()
print("=" * 80)
print("REFERENCE RECONCILIATION")
print("=" * 80)
print(reference_reconciliation.to_string(index=False))

print()
print("=" * 80)
print("FIXED-VOLUME / FIXED-DENSITY DIAGNOSTICS")
print("=" * 80)
print(geo_diag.to_string(index=False))

print()
print("=" * 80)
print("NESTED UNCERTAINTY DECOMPOSITION")
print("=" * 80)

show_cols = [
    "boundary_sd_m",
    "metric",
    "reference_value",
    "mean",
    "p05",
    "p50",
    "p95",
    "U_grade",
    "U_geo_raw",
    "U_geo_report",
    "R_G",
    "fixed_to_full_variance_ratio",
]

print(summary[show_cols].to_string(index=False))

print()
print("=" * 80)
print("GRADE ANISOTROPY")
print("=" * 80)
print(anisotropy.to_string(index=False))

print()
print("=" * 80)
print("OUTPUT DIRECTORY")
print("=" * 80)
print(OUT)

print()
print("Main outputs:")
print("  reference_reconciliation.csv")
print("  geometry_bias_diagnostics.csv")
print("  geological_realisation_summary.csv")
print("  real_case_scenario_summary.csv")
print("  convergence_summary.csv")
print("  grade_anisotropy.csv")
print("  directional_logcu_variograms.csv")
print("  real_case_metadata.json")
print("  Fig9.eps / Fig9.pdf / Fig9.tiff")
print("  Fig10.eps / Fig10.pdf / Fig10.tiff")
print("  Fig11.eps / Fig11.pdf / Fig11.tiff")
print("  Fig12.eps / Fig12.pdf / Fig12.tiff")
print("  FigS2.eps / FigS2.pdf / FigS2.tiff")
print()
print("analysis complete.")
