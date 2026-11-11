"""Patrón del pétalo Plan A: aro circular + vuelo de tela. Coordenadas de patrón (s, y) en metros.
s = distancia desde la raíz a lo largo del eje; y = transversal."""
import numpy as np
from scipy.spatial import Delaunay

P_EXP, Q_EXP = 0.7, 1.244


def half_width(s, length=4.0, width=3.5):
    u = np.clip(np.asarray(s, float) / length, 0, 1)
    up = P_EXP / (P_EXP + Q_EXP)
    g = u ** P_EXP * (1 - u) ** Q_EXP
    return 0.5 * width * g / (up ** P_EXP * (1 - up) ** Q_EXP)


_BASE = half_width


def _base_hw(s, length=4.0, width=3.5):
    return _BASE(s, length, width)


def rounded_tip(length, width, r_tip):
    """Lágrima con la punta redondeada por un arco de radio r_tip (la varilla no puede hacer un pico)."""
    def fit(Lb):
        ss = np.linspace(0.5 * Lb, Lb - 1e-4, 6000)
        w = _base_hw(ss, Lb, width)
        dw = np.gradient(w, ss)
        f = w * np.sqrt(1 + dw ** 2) - r_tip
        k = np.where(f[:-1] * f[1:] <= 0)[0][-1]
        s1 = ss[k]
        return s1, s1 + w[k] * dw[k], Lb
    lo, hi = length, length + 3 * r_tip
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s1, sc, _ = fit(mid)
        lo, hi = (mid, hi) if sc + r_tip < length else (lo, mid)
    s1, sc, Lb = fit(0.5 * (lo + hi))

    def hw(s):
        s = np.asarray(s, float)
        return np.where(s <= s1, _base_hw(np.minimum(s, s1), Lb, width), np.sqrt(np.maximum(r_tip ** 2 - (s - sc) ** 2, 0.0)))
    return hw


def build(length=4.0, width=3.5, hoop_r=1.0, h=0.12, n_hoop=60, clamp_s=0.035, r_tip=None):
    c = np.array([hoop_r, 0.0])
    if r_tip:
        _hw = rounded_tip(length, width, r_tip)
        global half_width
        _old = half_width
        half_width = lambda s, L=None, W=None: _hw(s)
        try:
            return build(length, width, hoop_r, h, n_hoop, clamp_s, None)
        finally:
            half_width = _old

    def in_drop(p):
        return (p[:, 0] >= 0) & (p[:, 0] <= length) & (np.abs(p[:, 1]) <= half_width(p[:, 0], length, width))

    def rdist(p):
        return np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])

    # aro
    phi = 2 * np.pi * np.arange(n_hoop) / n_hoop
    hoop = np.c_[hoop_r - hoop_r * np.cos(phi), hoop_r * np.sin(phi)]
    # contorno del vuelo (lágrima) fuera del disco
    ss = length * (1 - np.linspace(1, 0, 6001) ** 2)          # más denso hacia la punta
    ww = half_width(ss, length, width)
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(ss), np.diff(ww)))]
    n_b = int(round(arc[-1] / h))
    sb = np.interp(np.linspace(0, arc[-1], n_b + 1), arc, ss)
    up = np.c_[sb, half_width(sb, length, width)]
    lo = np.c_[sb[:-1], -half_width(sb[:-1], length, width)]
    bnd = np.r_[up, lo]
    bnd = bnd[rdist(bnd) > hoop_r + 0.45 * h]
    bnd = np.r_[bnd, [[length, 0.0]]] if not np.any(np.all(np.isclose(bnd, [length, 0.0]), axis=1)) else bnd
    # interior en malla hexagonal con fila sobre el eje
    rows = int(np.ceil(width / 2 / (h * np.sqrt(3) / 2))) + 1
    pts = []
    for j in range(-rows, rows + 1):
        y = j * h * np.sqrt(3) / 2
        off = 0.5 * h if j % 2 else 0.0
        xs = np.arange(-h + off, length + h, h)
        pts.append(np.c_[xs, np.full_like(xs, y)])
    grid = np.concatenate(pts)
    rd = rdist(grid)
    inside_disc = rd < hoop_r - 0.5 * h
    in_vuelo = in_drop(grid) & (rd > hoop_r + 0.5 * h) & (half_width(grid[:, 0], length, width) - np.abs(grid[:, 1]) > 0.45 * h) \
        & (grid[:, 0] < length - 0.45 * h)
    grid = grid[inside_disc | in_vuelo]
    P = np.r_[hoop, bnd, grid]
    # quitar duplicados cercanos
    keep = np.ones(len(P), bool)
    for i in range(n_hoop, len(P)):
        d = np.hypot(*(P[:i][keep[:i]] - P[i]).T)
        if d.size and d.min() < 0.35 * h:
            keep[i] = False
    P = P[keep]
    tri = Delaunay(P).simplices
    cen = P[tri].mean(1)
    ok = (rdist(cen) < hoop_r) | in_drop(cen)
    e = P[tri][:, [1, 2, 0]] - P[tri]
    ok &= np.hypot(e[..., 0], e[..., 1]).max(1) < 2.4 * h
    tri = tri[ok]
    # orientación antihoraria (normal +z en patrón)
    a, b, cc = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])
    tri[area2 < 0] = tri[area2 < 0][:, [0, 2, 1]]
    area = 0.5 * np.abs(area2)
    tri = tri[area > 1e-6]
    area = area[area > 1e-6]
    used = np.zeros(len(P), bool)
    used[tri.ravel()] = True
    assert used.all(), f'{(~used).sum()} nodos sueltos'
    # aristas
    E = np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)
    Eu, inv = np.unique(E, axis=0, return_inverse=True)
    adj_area = np.zeros(len(Eu))
    np.add.at(adj_area, inv.ravel(), np.tile(area, 3))
    hoop_idx = np.arange(n_hoop)
    hoop_edges = np.c_[hoop_idx, np.roll(hoop_idx, -1)]
    have = {tuple(x) for x in Eu}
    missing = [tuple(sorted(x)) for x in hoop_edges if tuple(sorted(x)) not in have]
    assert not missing, f'aristas del aro ausentes: {missing[:5]}'
    is_disc_node = rdist(P) < hoop_r - 1e-6
    is_disc_tri = rdist(P[tri].mean(1)) < hoop_r
    node_area = np.zeros(len(P))
    np.add.at(node_area, tri.ravel(), np.repeat(area / 3, 3))
    clamp = (P[:, 0] < clamp_s) & (rdist(P) <= hoop_r + 1e-6)
    axis_nodes = np.where(np.abs(P[:, 1]) < 1e-6)[0]
    axis_nodes = axis_nodes[np.argsort(P[axis_nodes, 0])]
    return dict(P=P, tri=tri, tri_area=area, edges=Eu, edge_area=adj_area, n_hoop=n_hoop, hoop_r=hoop_r,
                is_disc_node=is_disc_node, is_disc_tri=is_disc_tri, node_area=node_area, clamp=clamp,
                axis_nodes=axis_nodes, length=length, width=width, h=h, total_area=area.sum())


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = build()
    P = m['P']
    print('nodos', len(P), 'tri', len(m['tri']), 'aristas', len(m['edges']), 'área', round(m['total_area'], 3),
          'pinzados', int(m['clamp'].sum()), 'eje', len(m['axis_nodes']))
    L = np.hypot(*(P[m['edges'][:, 0]] - P[m['edges'][:, 1]]).T)
    print('arista min/med/max', L.min().round(3), np.median(L).round(3), L.max().round(3))
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.triplot(P[:, 0], P[:, 1], m['tri'], lw=0.4, color='#357')
    ax.plot(*P[np.r_[0:m['n_hoop'], 0]].T, color='#c22', lw=2)
    ax.plot(*P[m['clamp']].T, 'ko', ms=5)
    ax.plot(*P[m['axis_nodes']].T, 'g.', ms=4)
    ax.set_aspect('equal')
    fig.savefig('/home/claude/work/sim/mesh.png', dpi=80)
