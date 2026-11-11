"""Simulación física de un pétalo Plan A (aro elástico + tela + bisagra de raíz + elástico + rienda).
Unidades SI. Ejes del pétalo: x radial hacia fuera, y tangente (eje de bisagra), z arriba. Bisagra en el origen.
theta = ángulo del pétalo sobre la horizontal exterior: 180° recogido hacia el centro, 90° vertical, 0° abierto."""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
G = 9.81
EPS = 1e-12

STEEL = dict(E=205e9, Gs=80e9, rho=7850.0)
GFRP = dict(E=40e9, Gs=4e9, rho=1900.0)


def band(b_mm, t_mm, label=None):
    """Fleje de canto: cara ancha perpendicular al plano del pétalo (como en un reflector plegable)."""
    b, t = b_mm / 1e3, t_mm / 1e3
    return dict(label=label or f'Fleje 1070 {b_mm:g}×{t_mm:g} mm', kind='fleje', b_mm=b_mm, t_mm=t_mm,
                EIw=STEEL['E'] * b * t ** 3 / 12, EIs=STEEL['E'] * t * b ** 3 / 12, GJ=STEEL['Gs'] * b * t ** 3 / 3,
                lam=STEEL['rho'] * b * t, r_fiber_w=t / 2, E=STEEL['E'])


def rod(d_mm, label=None):
    d = d_mm / 1e3
    I = np.pi * d ** 4 / 64
    return dict(label=label or f'Varilla fibra de vidrio Ø{d_mm:g} mm', kind='varilla', d_mm=d_mm,
                EIw=GFRP['E'] * I, EIs=GFRP['E'] * I, GJ=GFRP['Gs'] * 2 * I, lam=GFRP['rho'] * np.pi * d ** 2 / 4,
                r_fiber_w=d / 2, E=GFRP['E'])


def tube_alu(d_mm, t_mm, label=None):
    do, di = d_mm / 1e3, (d_mm - 2 * t_mm) / 1e3
    I = np.pi * (do ** 4 - di ** 4) / 64
    return dict(label=label or f'Tubo de aluminio Ø{d_mm:g}×{t_mm:g} mm rolado', kind='tubo', d_mm=d_mm,
                EIw=70e9 * I, EIs=70e9 * I, GJ=26e9 * 2 * I, lam=2700 * np.pi * (do ** 2 - di ** 2) / 4, r_fiber_w=do / 2, E=70e9)


def _unit(v):
    return v / jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + EPS)


class Petal:
    def __init__(self, mesh, hoop, fabric_gsm=80.0, fabric_Et=3000.0, comp_ratio=0.02, pre_strain=0.002,
                 hoop_EA=2.0e4, ribs=(), rigid_nodes=None, rigid_mass=None, theta_open_deg=12.0, sector_r=0.15, ring_r=1.0,
                 spring_M0=20.0, spring_delta_deg=15.0, disc_contact_on=True, cinch=None, tray_z=-0.03, tray_r=1.12, dt=1e-4):
        self.mesh, self.hoop, self.ribs = mesh, hoop, list(ribs)
        self.dt = dt
        P = mesh['P']
        N = len(P)
        self.N = N
        self.P0 = jnp.asarray(np.c_[P, np.zeros(N)])            # patrón plano (s, y, 0)
        clamp = mesh['clamp'].copy()
        if rigid_nodes is not None:
            clamp[np.asarray(rigid_nodes)] = True
        self.clamp_np = clamp
        self.clamp = jnp.asarray(clamp)
        self.free = jnp.asarray(~clamp)[:, None]
        nh = mesh['n_hoop']
        self.nh = nh
        # --- tela
        e = mesh['edges']
        L0 = np.hypot(*(P[e[:, 0]] - P[e[:, 1]]).T)
        mid = 0.5 * (P[e[:, 0]] + P[e[:, 1]])
        in_disc = np.hypot(mid[:, 0] - mesh['hoop_r'], mid[:, 1]) < mesh['hoop_r'] - 1e-6
        is_hoop_edge = (e[:, 0] < nh) & (e[:, 1] < nh) & ((np.abs(e[:, 0] - e[:, 1]) == 1) | (np.abs(e[:, 0] - e[:, 1]) == nh - 1))
        self.e0, self.e1 = jnp.asarray(e[:, 0]), jnp.asarray(e[:, 1])
        self.eL0 = jnp.asarray(L0 * np.where(in_disc & ~is_hoop_edge, 1 - pre_strain, 1.0))
        self.ek = jnp.asarray(np.minimum(fabric_Et * mesh['edge_area'] / L0 ** 2, 1.4 * fabric_Et))
        self.comp_ratio = comp_ratio
        # --- aro
        hi = np.arange(nh)
        self.h_prev, self.h_i, self.h_next = jnp.asarray(np.roll(hi, 1)), jnp.asarray(hi), jnp.asarray(np.roll(hi, -1))
        hl = np.hypot(*(P[hi] - P[np.roll(hi, -1)]).T)
        self.hl = jnp.asarray(hl)
        self.hlbar = jnp.asarray(0.5 * (hl + np.roll(hl, 1)))
        self.hoop_k = hoop_EA / hl.mean()
        self.EIw, self.EIs, self.GJ = hoop['EIw'], hoop['EIs'], hoop['GJ']
        # triángulos interiores adyacentes a cada nodo del aro (para la normal de la funda)
        tri = mesh['tri']
        adj = [[] for _ in range(nh)]
        for k, t in enumerate(tri):
            if mesh['is_disc_tri'][k]:
                for n in t:
                    if n < nh:
                        adj[n].append(k)
        K = max(len(a) for a in adj)
        idx = np.zeros((nh, K), int)
        wgt = np.zeros((nh, K))
        for n, a in enumerate(adj):
            idx[n, :len(a)] = a
            wgt[n, :len(a)] = 1
        self.adj_tri, self.adj_w = jnp.asarray(tri[idx]), jnp.asarray(wgt)   # (nh, K, 3)
        self.adj_norm = jnp.asarray((2 * mesh['tri_area'][idx] * wgt).sum(1))
        self.hoop_clamped = jnp.asarray(clamp[:nh])
        # --- nervaduras elásticas (varillas): cadenas de nodos de la malla
        self.rib_data = []
        for rb in self.ribs:
            idx = np.asarray(rb['nodes'])
            if rb.get('closed'):
                idx = np.r_[idx, idx[0], idx[1]]
            rl = np.hypot(*(P[idx[1:]] - P[idx[:-1]]).T) * (1 + rb.get('pre', 0.0))
            self.rib_data.append((jnp.asarray(idx), jnp.asarray(rl), rb['mat']['EIw'], hoop_EA / rl.mean()))
        # --- masas
        m = fabric_gsm / 1e3 * mesh['node_area']
        m[:nh] += hoop['lam'] * 0.5 * (hl + np.roll(hl, 1)) * 1.08       # +8 % funda y costura
        for rb in self.ribs:
            idx = np.asarray(rb['nodes'])
            if rb.get('closed'):
                idx = np.r_[idx, idx[0]]
            seg = np.hypot(*(P[idx[1:]] - P[idx[:-1]]).T)
            mr = np.zeros(len(idx))
            mr[:-1] += seg / 2
            mr[1:] += seg / 2
            m[idx] += rb['mat']['lam'] * mr
        if rigid_mass is not None:
            m = m + np.asarray(rigid_mass)
        self.m_np = m
        self.m = jnp.asarray(np.maximum(m, 1.0e-3))[:, None]     # masa inercial mínima (estabilidad); el peso usa la real
        self.m_true = jnp.asarray(m)
        self.mass_total = float(m.sum())
        # --- contacto vuelo ↔ disco
        rd = np.hypot(P[:, 0] - mesh['hoop_r'], P[:, 1])
        self.vuelo_idx = jnp.asarray(np.where(rd > mesh['hoop_r'] + 1e-6)[0])
        dn = np.where(rd <= mesh['hoop_r'] + 1e-6)[0]
        self.disc_idx = jnp.asarray(dn)
        self.disc_sy = jnp.asarray(P[dn])
        self.hoop_r = mesh['hoop_r']
        # --- mecanismo
        self.th_open = np.deg2rad(theta_open_deg)
        self.sector_r, self.ring_r = sector_r, ring_r
        self.M0, self.delta = spring_M0, np.deg2rad(spring_delta_deg)
        self.th_taper = np.deg2rad(45.0)
        self.tray_z, self.tray_r = tray_z, tray_r
        self.disc_contact_on = disc_contact_on
        # cordón de contención: recoge puntos del contorno hacia un ancla en el brazo, por el envés
        self.cinch = cinch
        if cinch is not None:
            self.c_idx = jnp.asarray(np.asarray(cinch['nodes']))
            a = np.broadcast_to(np.asarray(cinch['anchor_sy'], float), (len(cinch['nodes']), 2)).copy()
            self.c_anchor = jnp.asarray(a)
            self.c_L0 = jnp.asarray(np.hypot(*(P[np.asarray(cinch['nodes'])] - a).T) + 0.03)
            self.c_Lmin = jnp.asarray(np.broadcast_to(np.asarray(cinch.get('Lmin', 0.15), float), (len(cinch['nodes']),)).copy())
            self.c_k = cinch.get('k', 1500.0)
        self.I_th = 0.03
        self.k_rope, self.k_stop = 2.0e4, 3.0e3
        self._grad = jax.grad(self.energy, argnums=(0, 1))
        self._chunk = jax.jit(self._run_chunk, static_argnums=(3,))

    # ---------- cinemática de la pieza de raíz
    def rigid(self, th):
        c, s = jnp.cos(th), jnp.sin(th)
        return jnp.stack([self.P0[:, 0] * c, self.P0[:, 1], self.P0[:, 0] * s], axis=1)

    @staticmethod
    def normal_A(th):
        return jnp.array([-jnp.sin(th), 0.0, jnp.cos(th)])

    # ---------- energía elástica + gravedad + apoyo
    def energy(self, x, th, lid=10.0, cz=1.0):
        x = jnp.where(self.free, x, self.rigid(th))
        # tela: resortes de arista, casi sin rigidez a compresión (se arruga)
        d = x[self.e1] - x[self.e0]
        L = jnp.sqrt(jnp.sum(d * d, axis=1) + EPS)
        dl = L - self.eL0
        E = 0.5 * jnp.sum(jnp.where(dl > 0, 1.0, self.comp_ratio) * self.ek * dl * dl)
        # aro: inextensible (penalizado)
        xp, xi, xn = x[self.h_prev], x[self.h_i], x[self.h_next]
        ep, en = xi - xp, xn - xi
        lp = jnp.sqrt(jnp.sum(ep * ep, axis=1) + EPS)
        ln = jnp.sqrt(jnp.sum(en * en, axis=1) + EPS)
        E += 0.5 * self.hoop_k * jnp.sum((ln - self.hl) ** 2)
        # normal de la tela en cada nodo del aro (la funda orienta el fleje). Regularizada: si la tela
        # se arruga junto al aro la normal pierde peso y el fleje queda solo con su eje débil.
        a, b, c = x[self.adj_tri[..., 0]], x[self.adj_tri[..., 1]], x[self.adj_tri[..., 2]]
        nn = jnp.sum(jnp.cross(b - a, c - a) * self.adj_w[..., None], axis=1) / self.adj_norm[:, None]
        nn = jnp.where(self.hoop_clamped[:, None], self.normal_A(th)[None, :], nn)
        t = _unit(ep + en)
        nn = nn - jnp.sum(nn * t, axis=1, keepdims=True) * t
        n = nn / jnp.sqrt(jnp.sum(nn * nn, axis=1, keepdims=True) + 0.15 ** 2) * 1.0112
        kap = (en / ln[:, None] - ep / lp[:, None]) / self.hlbar[:, None]   # κ·N, acotada aunque el aro se doble sobre sí
        k_out = jnp.sum(kap * n, axis=1)      # curvatura fuera del plano de la tela → eje fuerte del fleje de canto
        E += 0.5 * jnp.sum(self.hlbar * (self.EIw * jnp.sum(kap * kap, axis=1) + (self.EIs - self.EIw) * k_out ** 2))
        # torsión: giro de la normal alrededor del eje del fleje entre nodos consecutivos (acotada)
        te = en / ln[:, None]
        dn = n[self.h_next] - n
        dn = dn - jnp.sum(dn * te, axis=1, keepdims=True) * te
        E += 0.5 * self.GJ * jnp.sum(jnp.sum(dn * dn, axis=1) / self.hl)
        # nervaduras
        for idx, rl, EI, kst in self.rib_data:
            r = x[idx]
            e = r[1:] - r[:-1]
            le = jnp.sqrt(jnp.sum(e * e, axis=1) + EPS)
            tt = e / le[:, None]
            lb = 0.5 * (rl[:-1] + rl[1:])
            dT = tt[1:] - tt[:-1]
            E += 0.5 * EI * jnp.sum(jnp.sum(dT * dT, axis=1) / lb)
            E += 0.5 * kst * jnp.sum((le - rl) ** 2)
        if self.cinch is not None:
            E += 0.5 * self.c_k * jnp.sum(self.cinch_stretch(x, th, cz) ** 2)
        # gravedad
        E += G * jnp.sum(self.m_true * x[:, 2])
        # apoyo del paquete (bandeja sobre el bastidor)
        on_tray = (x[:, 0] + self.ring_r) ** 2 + x[:, 1] ** 2 < self.tray_r ** 2
        pen = jnp.where(on_tray, jnp.maximum(0.0, self.tray_z - x[:, 2]), 0.0)
        E += 0.5 * 3000.0 * jnp.sum(pen ** 2)
        # tapa: la capa de encima en la pila (solo mientras está guardado)
        pen2 = jnp.where(on_tray, jnp.maximum(0.0, x[:, 2] - lid), 0.0)
        E += 0.5 * 800.0 * jnp.sum(pen2 ** 2)
        return E

    def hoop_diag(self, x, th):
        x = jnp.where(self.free, x, self.rigid(th))
        xp, xi, xn = x[self.h_prev], x[self.h_i], x[self.h_next]
        ep, en = xi - xp, xn - xi
        lp = jnp.sqrt(jnp.sum(ep * ep, axis=1) + EPS)
        ln = jnp.sqrt(jnp.sum(en * en, axis=1) + EPS)
        a, b, c = x[self.adj_tri[..., 0]], x[self.adj_tri[..., 1]], x[self.adj_tri[..., 2]]
        nn = jnp.sum(jnp.cross(b - a, c - a) * self.adj_w[..., None], axis=1)
        nn = jnp.where(self.hoop_clamped[:, None], self.normal_A(th)[None, :], nn)
        t = _unit(ep + en)
        n = _unit(nn - jnp.sum(nn * t, axis=1, keepdims=True) * t)
        kap = (en / ln[:, None] - ep / lp[:, None]) / self.hlbar[:, None]
        k_out = jnp.sum(kap * n, axis=1)
        k_in = jnp.sqrt(jnp.maximum(jnp.sum(kap * kap, axis=1) - k_out ** 2, 0.0))
        te = en / ln[:, None]
        dn = n[self.h_next] - n
        dn = dn - jnp.sum(dn * te, axis=1, keepdims=True) * te
        tw = jnp.sqrt(jnp.sum(dn * dn, axis=1)) / self.hl
        return k_in, k_out, tw

    def cinch_stretch(self, x, th, cz):
        a = self.c_anchor
        nA = self.normal_A(th)
        anchor = jnp.stack([a[:, 0] * jnp.cos(th), a[:, 1], a[:, 0] * jnp.sin(th)], axis=1) - 0.06 * nA[None, :]
        d = x[self.c_idx] - anchor
        dist = jnp.sqrt(jnp.sum(d * d, axis=1) + EPS)
        L = self.c_Lmin + (self.c_L0 - self.c_Lmin) * cz
        return jnp.maximum(0.0, dist - L)

    def disc_contact(self, x, v, th, gap=0.008, k=1200.0, c=1.5):
        """La tela del vuelo no puede atravesar el disco: se mantiene del lado B (envés)."""
        nA = self.normal_A(th)
        u = jnp.array([jnp.cos(th), 0.0, jnp.sin(th)])
        xv = x[self.vuelo_idx]
        sv, yv, hv = xv @ u, xv[:, 1], xv @ nA
        inside = (sv - self.hoop_r) ** 2 + yv ** 2 < (self.hoop_r - 0.02) ** 2
        d2 = (sv[:, None] - self.disc_sy[None, :, 0]) ** 2 + (yv[:, None] - self.disc_sy[None, :, 1]) ** 2
        w = jax.nn.softmax(-d2 / 0.07 ** 2, axis=1)                 # referencia suave: sin saltos entre nodos vecinos
        hd = x[self.disc_idx] @ nA
        href = w @ hd
        pen = jnp.where(inside, jnp.maximum(0.0, hv - (href - gap)), 0.0)
        rel = v[self.vuelo_idx] @ nA - w @ (v[self.disc_idx] @ nA)
        f = jnp.where(pen > 0, jnp.maximum(0.0, k * pen + c * rel), 0.0)
        F = jnp.zeros_like(x)
        F = F.at[self.vuelo_idx].add(-f[:, None] * nA[None, :])
        F = F.at[self.disc_idx].add((f @ w)[:, None] * nA[None, :])
        return F

    def spring_torque(self, th):
        taper = jnp.clip(0.5 + 0.5 * (th - self.th_open) / (self.th_taper - self.th_open), 0.5, 1.0)
        return self.M0 * jnp.cos(th + self.delta) * taper

    def _step(self, state, ctrl):
        x, v, th, om = state
        th_p, kin, u_rope, spr, damp, lid, cz = ctrl
        gx, gth = self._grad(x, th, lid, cz)
        F = -gx + (self.disc_contact(x, v, th) if self.disc_contact_on else 0.0)
        v = (v + self.dt * F / self.m) * jnp.exp(-damp * self.dt)
        v = jnp.where(self.free, v, 0.0)
        x = x + self.dt * v
        # bisagra
        stretch = self.sector_r * (jnp.pi - th) - u_rope
        T = jnp.where(stretch > 0, self.k_rope * stretch + 40.0 * jnp.maximum(0.0, -om * self.sector_r), 0.0)
        Ms = spr * self.spring_torque(th)
        M_stop = self.k_stop * jnp.maximum(0.0, self.th_open - th) - self.k_stop * jnp.maximum(0.0, th - jnp.pi)
        Q = -gth + Ms + T * self.sector_r + M_stop
        om_d = (om + self.dt * Q / self.I_th) * jnp.exp(-8.0 * self.dt)
        th_d = th + self.dt * om_d
        th = jnp.where(kin > 0.5, th_p, th_d)
        om = jnp.where(kin > 0.5, 0.0, om_d)
        x = jnp.where(self.free, x, self.rigid(th))
        Tc = self.c_k * jnp.max(self.cinch_stretch(x, th, cz)) if self.cinch is not None else 0.0
        return (x, v, th, om), jnp.array([T, -gth, Ms, M_stop, Tc])

    def _run_chunk(self, state, c0, c1, n):
        def body(carry, i):
            f = (i + 1.0) / n
            return self._step(carry, c0 + (c1 - c0) * f)
        state, out = jax.lax.scan(body, state, jnp.arange(n))
        return state, out.mean(axis=0), out[-1]

    def run(self, state, c0, c1, seconds, fps=20, record=None, n_fixed=None):
        """Avanza interpolando el control de c0 a c1. ctrl = [theta_prescrita, cinemático, cuerda_u, resorte, amortiguación]."""
        frames = max(1, int(round(seconds * fps)))
        n = n_fixed or int(round(1.0 / fps / self.dt))
        c0 = jnp.asarray(list(c0) + [1.0] * (7 - len(c0)), float)
        c1 = jnp.asarray(list(c1) + [1.0] * (7 - len(c1)), float)
        for k in range(frames):
            a = c0 + (c1 - c0) * (k / frames)
            b = c0 + (c1 - c0) * ((k + 1) / frames)
            state, mean, last = self._chunk(state, a, b, n)
            if record is not None:
                record(state, np.asarray(mean), np.asarray(last))
        return state

    def init_state(self, th):
        return (self.rigid(jnp.asarray(th)), jnp.zeros((self.N, 3)), jnp.asarray(float(th)), jnp.asarray(0.0))
