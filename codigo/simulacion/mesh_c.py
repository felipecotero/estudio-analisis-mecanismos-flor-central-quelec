"""Patrón de pétalo de la opción C: uña angosta desde el centro hasta el aro, lámina ovalada desde el aro.
Coordenadas de patrón (s, y): s desde la raíz junto al centro; la bisagra está en s = SH."""
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path

SH = 0.75          # largo de la uña: raíz a 0,25 m del centro, bisagra en el aro de 1,0 m
CLAW0, CLAW1 = 0.07, 0.20


def half_width(s, blade_len, width):
    s = np.asarray(s, float)
    u = np.clip((s - SH) / blade_len, 0, 1)
    a, b = 1.0, 0.55
    up = a / (a + b)
    g = u ** a * (1 - u) ** b / (up ** a * (1 - up) ** b)
    blade = CLAW1 + (width / 2 - CLAW1) * g
    blade = np.maximum(blade, CLAW1 * (1 - u) ** 0.5)          # sin estrangular junto a la bisagra
    claw = CLAW0 + (CLAW1 - CLAW0) * np.clip(s / SH, 0, 1) ** 1.5
    return np.where(s <= SH, claw, blade)


def build(blade_len=3.25, width=3.5, h=1 / 6):
    L = SH + blade_len
    # contorno
    ss = np.r_[np.linspace(0, SH, 6)[:-1], SH + blade_len * (1 - np.linspace(1, 0, 3000) ** 2)]
    ww = half_width(ss, blade_len, width)
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(ss), np.diff(ww)))]
    n_b = int(round(arc[-1] / h))
    sb = np.interp(np.linspace(0, arc[-1], n_b + 1), arc, ss)
    sb = np.unique(np.r_[sb, SH])                               # nodo exacto en la bisagra
    wb = half_width(sb, blade_len, width)
    upper = np.c_[sb, wb]
    lower = np.c_[sb[::-1][1:-1], -wb[::-1][1:-1]]
    tip = np.array([[L, 0.0]])
    if upper[-1, 1] > 1e-6:
        upper = np.r_[upper, tip]
    else:
        upper[-1] = tip
    root = np.array([[0.0, 0.0]])
    loop = np.r_[root, upper[1:] if upper[0, 1] < 1e-6 else upper, lower[::-1][::-1]]
    loop = np.r_[upper, lower]                                  # de la raíz (arriba) a la punta y de vuelta
    poly = Path(np.r_[loop, loop[:1]])
    # interior: rejilla hexagonal alineada con la bisagra (filas pares con nodos en s' = k·h)
    dy = h * np.sqrt(3) / 2
    pts = []
    for j in range(-int(width / 2 / dy) - 2, int(width / 2 / dy) + 3):
        off = 0.5 * h if j % 2 else 0.0
        xs = SH + np.arange(-int(SH / h) - 1, int(blade_len / h) + 2) * h + off
        pts.append(np.c_[xs, np.full_like(xs, j * dy)])
    grid = np.concatenate(pts)
    inside = poly.contains_points(grid, radius=-0.4 * h) & (grid[:, 0] > 0.3 * h)
    grid = grid[inside]
    # lejos del borde
    d = np.min(np.hypot(grid[:, None, 0] - loop[None, :, 0], grid[:, None, 1] - loop[None, :, 1]), axis=1)
    grid = grid[d > 0.45 * h]
    P = np.r_[loop, grid]
    nb = len(loop)
    tri = Delaunay(P).simplices
    cen = P[tri].mean(1)
    tri = tri[poly.contains_points(cen)]
    e = P[tri][:, [1, 2, 0]] - P[tri]
    tri = tri[np.hypot(e[..., 0], e[..., 1]).max(1) < 2.2 * h]
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    ar2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tri[ar2 < 0] = tri[ar2 < 0][:, [0, 2, 1]]
    area = 0.5 * np.abs(ar2)
    keep = area > 1e-5
    tri, area = tri[keep], area[keep]
    used = np.zeros(len(P), bool); used[tri.ravel()] = True
    assert used.all(), f'{(~used).sum()} nodos sueltos'
    E = np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)
    Eu = np.unique(E, axis=0)
    eA = np.zeros(len(Eu))
    lut = {tuple(k): i for i, k in enumerate(Eu)}
    for t, A in zip(tri, area):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            eA[lut[tuple(sorted((t[i], t[j])))]] += A
    nA = np.zeros(len(P)); np.add.at(nA, tri.ravel(), np.repeat(area / 3, 3))
    sp = P[:, 0] - SH
    on = lambda cond: np.where(cond)[0]
    arm = on((np.abs(P[:, 1]) < 1e-6) & (sp > 1e-6) & (sp <= 2.0 + 1e-6))
    arm = arm[np.argsort(P[arm, 0])]
    spreader = on((np.abs(sp - 1.0) < 1e-6) & (np.abs(P[:, 1]) <= 1.0 + 1e-6))
    spreader = spreader[np.argsort(P[spreader, 1])]
    joint = spreader[np.abs(P[spreader, 1]) <= 0.3]
    antenna = on((np.abs(P[:, 1]) < 1e-6) & (sp >= 2.0 - 2 * h - 1e-6))
    antenna = antenna[np.argsort(P[antenna, 0])]
    # flex del contorno: borde de la lámina, de bisagra a bisagra pasando por la punta
    hinge_up = int(np.where((np.abs(loop[:, 0] - SH) < 1e-9) & (loop[:, 1] > 0))[0][0])
    hinge_lo = int(np.where((np.abs(loop[:, 0] - SH) < 1e-9) & (loop[:, 1] < 0))[0][0])
    perimeter = np.r_[np.arange(hinge_up, nb), np.arange(0, hinge_lo + 1)] if hinge_lo < hinge_up else np.arange(hinge_up, hinge_lo + 1)
    tip_i = int(np.argmin(np.hypot(P[:, 0] - L, P[:, 1])))
    spk = SH + blade_len * 0.55
    side_up = int(np.argmin(np.hypot(P[:nb, 0] - spk, P[:nb, 1] - width / 2)))
    side_lo = int(np.argmin(np.hypot(P[:nb, 0] - spk, P[:nb, 1] + width / 2)))
    fixed = sp <= 1e-6
    rigid = np.zeros(len(P), bool); rigid[arm] = True; rigid[joint] = True
    return dict(P=P, tri=tri, tri_area=area, edges=Eu, edge_area=eA, node_area=nA, L=L, width=width, blade_len=blade_len, h=h,
                fixed=fixed, rigid=rigid & ~fixed, arm=arm, spreader=spreader, antenna=antenna, perimeter=perimeter,
                cinch=np.array([tip_i, side_up, side_lo]), loop=np.arange(nb), total_area=area.sum())


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, kw in (('normal', {}), ('grande', dict(blade_len=3.6, width=3.9))):
        m = build(**kw)
        P = m['P']
        print(name, 'nodos', len(P), 'tri', len(m['tri']), 'área', round(m['total_area'], 2), 'largo', round(m['L'], 2), 'ancho', round(np.ptp(P[:, 1]), 2),
              'fijos', m['fixed'].sum(), 'rígidos', m['rigid'].sum(), 'flex', len(m['perimeter']), 'antena', len(m['antenna']), 'travesaño', len(m['spreader']))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.triplot(P[:, 0], P[:, 1], m['tri'], lw=0.4)
        ax.plot(*P[m['perimeter']].T, 'g-', lw=2); ax.plot(*P[m['arm']].T, 'k-', lw=3); ax.plot(*P[m['spreader']].T, 'r-', lw=2)
        ax.plot(*P[m['antenna']].T, 'y-', lw=2); ax.plot(*P[m['fixed']].T, 'b.', ms=5); ax.plot(*P[m['cinch']].T, 'mo', ms=9)
        ax.set_aspect('equal'); fig.savefig(f'/home/claude/work/sim/meshC_{name}.png', dpi=60)
