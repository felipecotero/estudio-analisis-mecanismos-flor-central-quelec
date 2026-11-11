"""Configuraciones del pétalo Plan A."""
import numpy as np
import mesh as M
from petal import Petal, band, rod, GFRP

ALU_TUBE_25 = dict(label='Tubo de aluminio Ø25×1,5 mm', EIw=70e9 * np.pi * (25 ** 4 - 22 ** 4) / 64 * 1e-12,
                   lam=2700 * np.pi * (25 ** 2 - 22 ** 2) / 4 * 1e-6)


def nodes_on(P, s=None, y=None, lo=-1e9, hi=1e9):
    if s is not None:
        idx = np.where((np.abs(P[:, 0] - s) < 1e-6) & (P[:, 1] >= lo - 1e-6) & (P[:, 1] <= hi + 1e-6))[0]
        return idx[np.argsort(P[idx, 1])]
    idx = np.where((np.abs(P[:, 1] - y) < 1e-6) & (P[:, 0] >= lo - 1e-6) & (P[:, 0] <= hi + 1e-6))[0]
    return idx[np.argsort(P[idx, 0])]


def boundary_loop(m):
    """Nodos del contorno del patrón, en orden, empezando en la raíz."""
    tri = m['tri']
    E = np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)
    Eu, cnt = np.unique(E, axis=0, return_counts=True)
    nb = {}
    for a, b in Eu[cnt == 1]:
        nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    loop = [0, max(nb[0], key=lambda k: m['P'][k, 1])]
    while True:
        nxt = [k for k in nb[loop[-1]] if k != loop[-2]][0]
        if nxt == 0:
            break
        loop.append(nxt)
    return np.array(loop)


def make(config, hoop=None, mesh=None, **kw):
    """config: 'literal' (aro + vuelo), 'cruceta' (+ brazo rígido y travesaño), 'antena' (+ varilla hasta la punta)."""
    m = mesh or (M.build(h=0.125, r_tip=kw.pop('r_tip', 0.5)) if config == 'flex' else M.build(h=0.125))
    P = m['P']
    hoop = hoop or band(15, 0.8)
    if config == 'literal':
        kw.pop('antenna', None); kw.pop('spreader', None)
        return Petal(m, hoop, disc_contact_on=False, **kw), m
    arm = nodes_on(P, y=0.0, lo=0.0, hi=2.0)                # brazo: raíz → borde lejano del aro
    joint = nodes_on(P, s=1.0, lo=-0.22, hi=0.22)           # pieza de cruce
    rigid = np.unique(np.r_[arm, joint])
    rm = np.zeros(len(P))
    seg = np.diff(P[arm, 0])
    rm[arm[:-1]] += ALU_TUBE_25['lam'] * seg / 2
    rm[arm[1:]] += ALU_TUBE_25['lam'] * seg / 2
    rm[arm[0]] += 0.25                                        # pieza de raíz y sector
    ribs = [dict(name='Travesaño', nodes=nodes_on(P, s=1.0, lo=-1.0, hi=1.0), mat=kw.pop('spreader', rod(8)))]
    if config == 'flex':
        loop = boundary_loop(m)
        # una sola varilla: sus dos extremos se juntan en un bolsillo en la punta (ahí no hay flexión)
        ribs.append(dict(name='Flex de contorno', nodes=loop, closed=True, pre=kw.pop('pre', 0.01), mat=kw.pop('perimeter', rod(8))))
        ant = kw.pop('antenna', None)
        if ant is not None:
            ribs.append(dict(name='Antena de punta', nodes=nodes_on(P, y=0.0, lo=1.875, hi=4.0), mat=ant))
        pts = kw.pop('cinch_points', [(4.0, 0.0), (1.4, 1.75), (1.4, -1.75)])
        cn = [int(loop[np.argmin(np.hypot(P[loop, 0] - a, P[loop, 1] - b))]) for a, b in pts]
        kw['cinch'] = dict(nodes=cn, anchor_sy=kw.pop('cinch_anchors', (1.0, 0.0)), Lmin=kw.pop('cinch_Lmin', [0.15, 0.5, 0.5]))
        return Petal(m, hoop, ribs=ribs, rigid_nodes=rigid, rigid_mass=rm, **kw), m
    if config == 'antena':
        tip = nodes_on(P, y=0.0, lo=1.875, hi=4.0)
        ribs.append(dict(name='Antena de punta', nodes=tip, mat=kw.pop('antenna', rod(8))))
    else:
        kw.pop('antenna', None)
    return Petal(m, hoop, ribs=ribs, rigid_nodes=rigid, rigid_mass=rm, **kw), m


def make_B(**kw):
    """Opción B elegida: flex de contorno y antena de varilla de 6 mm; la punta se recoge hacia un lado del aro."""
    return make('flex', perimeter=rod(6), antenna=rod(6), pre=0.01, cinch_anchors=[[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]], cinch_Lmin=[0.5, 0.5, 0.5], **kw)
