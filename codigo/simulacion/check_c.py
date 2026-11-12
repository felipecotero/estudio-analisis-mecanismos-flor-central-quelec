"""Cuenta cruces arista–triángulo entre pétalos distintos (y de un pétalo consigo mismo, lejos en el patrón) en cada cuadro."""
import numpy as np, jax.numpy as jnp
from scipy.spatial import cKDTree
from sim_c import Flower
f = Flower()
d = np.load('anim_c.npz'); F = d['frames']
E = np.unique(np.sort(np.r_[f.tri_np[:, [0, 1]], f.tri_np[:, [1, 2]], f.tri_np[:, [2, 0]]], 1), axis=0)
eP = f.pid_np[E[:, 0]]; tP = f.tri_pid
def crossings(g):
    a, b = g[E[:, 0]], g[E[:, 1]]
    mid = 0.5 * (a + b); cen = g[f.tri_np].mean(1)
    pairs = cKDTree(mid).query_ball_tree(cKDTree(cen), 0.3)
    ei = np.repeat(np.arange(len(E)), [len(p) for p in pairs]); tj = np.concatenate([np.array(p, int) for p in pairs])
    same = eP[ei] == tP[tj]
    farP = np.hypot(*(f.P[E[ei]].mean(1) - f.P[f.tri_np[tj]].mean(1)).T) > 0.4
    ok = (~same | farP) & ~np.any(np.isin(f.tri_np[tj], E[ei]).reshape(len(tj), 3) if False else (f.tri_np[tj] == E[ei, 0:1]) | (f.tri_np[tj] == E[ei, 1:2]), axis=1)
    ei, tj = ei[ok], tj[ok]
    p0, p1 = g[E[ei, 0]], g[E[ei, 1]]
    A, B, C = (g[f.tri_np[tj, i]] for i in range(3))
    n = np.cross(B - A, C - A)
    d0 = np.einsum('ij,ij->i', p0 - A, n); d1 = np.einsum('ij,ij->i', p1 - A, n)
    s = d0 * d1 < 0
    t = d0[s] / (d0[s] - d1[s]); q = p0[s] + t[:, None] * (p1[s] - p0[s])
    A, B, C, n = A[s], B[s], C[s], n[s]
    c1 = np.einsum('ij,ij->i', np.cross(B - A, q - A), n) >= 0
    c2 = np.einsum('ij,ij->i', np.cross(C - B, q - B), n) >= 0
    c3 = np.einsum('ij,ij->i', np.cross(A - C, q - C), n) >= 0
    hit = (c1 & c2 & c3)
    ee, tt = ei[s][hit], tj[s][hit]
    return len(ee), int(np.sum(eP[ee] != tP[tt]))
if __name__ == '__main__':
    import sys
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = []
    for k in range(0, len(F), step):
        g = np.asarray(f.to_global(jnp.asarray(F[k], float)))
        n_all, n_other = crossings(g)
        out.append((d['rec'][k, 0], n_all, n_other))
    out = np.array(out)
    np.save('cross_c.npy', out)
    for r in out[::max(1, len(out) // 25)]: print(f't={r[0]:5.1f} cruces={int(r[1])} (entre pétalos distintos {int(r[2])})')
    print('máx cruces', int(out[:, 1].max()), 'cuadros con algún cruce', int((out[:, 1] > 0).sum()), 'de', len(out))
