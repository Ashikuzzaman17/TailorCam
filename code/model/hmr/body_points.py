"""
Tape-style measurements on an SMPL rest-pose (T-pose) mesh.

customBodyPoints.txt (from the HMR body-measurement project) marks vertex
rings on SMPL for each body part. Summing the ring's edge lengths, as the
original script did, zig-zags across the surface (the chest ring sums to
~146 cm on an average body, which the original code patched with a 0.69
fudge factor). Here the rings are used only to locate *where* to measure;
the value is what a tailor's tape would read:

  circumference = perimeter of the convex hull of the mesh cross-section
                  (a tape bridges hollows, exactly like a convex hull)
  lengths       = surface arcs / straight lines between body landmarks

Mesh axes (SMPL): +Y up, +X subject's left, +Z forward. Units: metres in,
centimetres out.
"""

import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

_POINTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'customBodyPoints.txt')
_UP = np.array([0.0, 1.0, 0.0])
_TORSO_HALF_W = 0.28     # T-pose arms are above chest level; legs are within this
_NIPPLES = (3042, 6489)  # SMPL vertex ids, left / right
_SHOULDERS = (1861, 6470)  # acromion: top of the shoulder over the joint, left / right
_NAVEL = 3501


def load_rings(path=_POINTS_FILE):
    """{'height': [...], 'waist': [...], 'belly': [...], ...} vertex indices."""
    rings, cur = {}, None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('#'):
                if 'DOUBLE CHECK' in s:
                    continue
                cur = s[1:].strip().lower().replace(' ', '_')
                rings[cur] = []
            elif cur and len(s.split()) >= 3:
                rings[cur].append(int(s.split()[1]))
    return rings


# ── Mesh slicing ──────────────────────────────────────────────────────────────
def _slice(V, F, origin, normal):
    """Intersect mesh with a plane → list of contour components (each N x 3)."""
    d = (V - origin) @ normal
    s = d[F] > 0
    cnt = s.sum(1)
    tri = F[(cnt == 1) | (cnt == 2)]
    if len(tri) == 0:
        return []
    # The 3 edges of every crossing triangle; exactly 2 of them cross the plane
    a = tri[:, [0, 1, 2]].ravel()
    b = tri[:, [1, 2, 0]].ravel()
    crosses = (d[a] > 0) != (d[b] > 0)
    a, b = a[crosses], b[crosses]
    keys = np.minimum(a, b).astype(np.int64) * len(V) + np.maximum(a, b)
    uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    ea, eb = a[first], b[first]
    t = (d[ea] / (d[ea] - d[eb]))[:, None]
    pts = V[ea] + t * (V[eb] - V[ea])
    # Each triangle contributes one segment joining its two crossing edges
    seg = inv.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(seg)), (seg[:, 0], seg[:, 1])),
                       shape=(len(uniq), len(uniq)))
    n, labels = connected_components(graph, directed=False)
    return [pts[labels == i] for i in range(n)]


def _plane_basis(normal):
    a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 0, 1.0])
    u = np.cross(normal, a); u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _hull_perimeter(points, normal):
    if len(points) < 3:
        return 0.0
    u, v = _plane_basis(normal)
    p2 = np.c_[points @ u, points @ v]
    hull = p2[ConvexHull(p2).vertices]
    return float(np.linalg.norm(np.diff(np.vstack([hull, hull[:1]]), axis=0), axis=1).sum())


def _torso_girth(V, F, y):
    """Horizontal girth at height y, everything within the torso/leg width."""
    comps = _slice(V, F, np.array([0, y, 0]), _UP)
    if not comps:
        return 0.0
    pts = np.vstack(comps)
    pts = pts[np.abs(pts[:, 0]) < _TORSO_HALF_W]
    return _hull_perimeter(pts, _UP)


def _ring_girth(V, F, ring, vertical=False):
    """Girth of the limb/neck the ring sits on (plane fitted to the ring)."""
    p = V[ring]
    c = p.mean(0)
    if vertical:
        n = _UP
    else:
        n = np.linalg.svd(p - c)[2][-1]          # least-variance direction
    comps = _slice(V, F, c, n)
    if not comps:
        return 0.0
    best = min(comps, key=lambda q: np.linalg.norm(q.mean(0) - c))
    return _hull_perimeter(best, n)


def _scan(V, F, y0, y1, pick, step=0.005):
    ys = np.arange(y0, y1 + 1e-9, step)
    g = np.array([_torso_girth(V, F, y) for y in ys])
    i = int(np.argmax(g) if pick == 'max' else np.argmin(np.where(g > 0, g, np.inf)))
    return g[i], ys[i]


def _lower_hull_length(pts2):
    """
    Length of the lower convex chain of 2-D points (sorted along axis 0,
    minimising axis 1) — i.e. a tape pulled taut over the surface from the
    first point to the last, bridging any hollows.
    """
    p = pts2[np.lexsort((pts2[:, 1], pts2[:, 0]))]
    chain = []
    for q in p:
        while len(chain) >= 2:
            o, a = chain[-2], chain[-1]
            if (a[0] - o[0]) * (q[1] - o[1]) - (a[1] - o[1]) * (q[0] - o[0]) <= 0:
                chain.pop()
            else:
                break
        chain.append(q)
    chain = np.array(chain)
    return float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())


def _shoulder_point(V, seed):
    """
    Shoulder point (acromion) = the outer corner of the shoulder seen from the
    front: where the shoulder's top line turns down into the arm. Searched
    around the seed vertex so it stays correct when the arm is lowered (the
    seed itself is skinned to the arm and slides off the corner).
    """
    s = np.sign(V[seed, 0])
    m = ((V[:, 1] > V[seed, 1] - 0.15) & (V[:, 1] < V[seed, 1] + 0.06)
         & (np.abs(V[:, 2] - V[seed, 2]) < 0.05) & (s * V[:, 0] > 0.10))
    idx = np.where(m)[0]
    if len(idx) == 0:
        return V[seed]
    return V[idx[np.argmax(s * V[idx, 0] + V[idx, 1])]]


def _back_arc(V, F, y_top, y_bottom):
    """Length down the centre back (x=0 plane) between two heights, tape-style."""
    comps = _slice(V, F, np.zeros(3), np.array([1.0, 0, 0]))
    if not comps:
        return abs(y_top - y_bottom)
    pts = np.vstack(comps)
    pts = pts[(pts[:, 1] <= y_top + 1e-3) & (pts[:, 1] >= y_bottom - 1e-3)
              & (pts[:, 2] < np.median(pts[:, 2]))]          # posterior half
    if len(pts) < 3:
        return abs(y_top - y_bottom)
    return _lower_hull_length(np.c_[pts[:, 1], pts[:, 2]])   # along y, min z


# ── Public API ────────────────────────────────────────────────────────────────
def measure(V, F, height_cm, rings=None, V_arms_down=None):
    """
    V: 6890 x 3 rest-pose (T-pose) vertices (metres), F: faces,
    height_cm: true height. V_arms_down: same body with the arms lowered
    (HMRBody.arms_down_mesh) — used for shoulder width and sleeve, which a
    tailor measures with the arms hanging; falls back to V if not given.
    Returns {key: cm} using TailorCam's measurement keys plus HMR extras.
    """
    rings = rings or load_rings()
    V = np.asarray(V, np.float64)
    VA = V if V_arms_down is None else np.asarray(V_arms_down, np.float64)
    F = np.asarray(F, np.int64)
    mesh_h = V[:, 1].max() - V[:, 1].min()
    k = float(height_cm / (mesh_h * 100.0))       # cm per mesh-cm
    cm = lambda m: round(float(m) * 100.0 * k, 1)

    y = {n: V[r][:, 1].mean() for n, r in rings.items()}

    # Torso girths: chest at nipple level (under the arms), waist at the
    # narrowest around the navel, hip at the fullest over the seat.
    y_nip = V[list(_NIPPLES)][:, 1].mean()
    chest, _ = _scan(V, F, y_nip - 0.015, y_nip + 0.015, 'max')
    waist, _ = _scan(V, F, y['belly'] - 0.04, y['belly'] + 0.06, 'min')
    hip, y_hip = _scan(V, F, y['hips'] - 0.06, y['hips'] + 0.04, 'max')
    belly = _torso_girth(V, F, y['belly'])

    # Shoulder width = tape across the back from shoulder point to shoulder point
    p_a, p_b = _shoulder_point(VA, _SHOULDERS[0]), _shoulder_point(VA, _SHOULDERS[1])
    y_sh = (p_a[1] + p_b[1]) / 2
    comps = _slice(VA, F, np.array([0, y_sh, 0]), _UP)
    pts = np.vstack(comps) if comps else np.zeros((0, 3))
    lo, hi = sorted([p_a[0], p_b[0]])
    pts = pts[(pts[:, 0] > lo) & (pts[:, 0] < hi) & (pts[:, 2] < max(p_a[2], p_b[2]))]
    xz = np.vstack([[p_a[0], p_a[2]], [p_b[0], p_b[2]], np.c_[pts[:, 0], pts[:, 2]]])
    shoulder = _lower_hull_length(xz)                       # across the back

    # Sleeve: shoulder point → wrist centre on the same side (arm hangs straight)
    wrist_c = VA[rings['wrist']].mean(0)
    sh_pt = p_a if np.sign(p_a[0]) == np.sign(wrist_c[0]) else p_b
    sleeve = np.linalg.norm(wrist_c - sh_pt)

    # Back length: back-neck (C7) → waist at navel level, along the spine
    neck = V[rings['neck']]
    c7 = neck[np.argmin(neck[:, 2])]
    back_len = _back_arc(V, F, c7[1], V[_NAVEL, 1])
    shirt_len = _back_arc(V, F, c7[1], y_hip - 0.04)  # regular shirt covers the seat

    return {
        'height':         round(float(height_cm), 1),
        'shoulder_width': cm(shoulder),
        'chest':          cm(chest),
        'waist':          cm(waist),
        'hip':            cm(hip),
        'neck':           cm(_ring_girth(V, F, rings['neck'])),
        'sleeve_length':  cm(sleeve),
        'back_length':    cm(back_len),
        'shirt_length':   cm(shirt_len),
        'belly':          cm(belly),
        'thigh':          cm(_ring_girth(V, F, rings['thigh'], vertical=True)),
        'wrist':          cm(_ring_girth(V, F, rings['wrist'])),
        'ankle':          cm(_ring_girth(V, F, rings['ankle'])),
    }
