"""Opción C8: pétalos que nacen del centro, contorno de una sola varilla que se pliega en ocho (como un reflector),
sin brazo rígido; abiertos los sostienen tres tensores desde un mástil central. Sin contacto entre telas."""
import numpy as np
import jax
import jax.numpy as jnp
import mesh_c as MC
from sim_c import PETALS, RING_R, HINGE_Z

jax.config.update('jax_enable_x64', True)
G, EPS = 9.81, 1e-12
MAST_H = 1.5                      # punta del mástil sobre la bisagra


def rod_mat(d_mm):
    d = d_mm / 1e3
    return dict(EI=40e9 * np.pi * d ** 4 / 64, lam=1900 * np.pi * d ** 2 / 4, d=d)


def coil_targets(m, chain, gap=0.025):
    """Tres aros apilados que pasan por los dos puntos de bisagra; misma longitud que la varilla."""
    P = m['P']
    q = np.c_[P[chain, 0] - MC.SH, P[chain, 1]]
    seg = np.hypot(*np.diff(q, axis=0).T)
    u = np.r_[0, np.cumsum(seg)]; Lp = u[-1]; u /= Lp
    h0 = abs(q[0, 1])
    R = Lp / (6 * np.pi)
    for _ in range(60):
        d = np.arcsin(h0 / R)
        R = Lp / (6 * np.pi - 2 * d)
    d = np.arcsin(h0 / R); X0 = np.sqrt(R * R - h0 * h0)
    phi_up = np.pi - d
    span = 6 * np.pi - 2 * d
    phi = phi_up - span * u
    C = np.c_[X0 + R * np.cos(phi), R * np.sin(phi), gap * (span * u) / (2 * np.pi)]
    return np.c_[q, np.zeros(len(q))], C, R, Lp


def eight_path(O, C, t):
    """Trayectoria de plegado: torcer la mitad lejana 180° (ocho) y luego recoger en tres aros."""
    sm = lambda x: (lambda c: c * c * (3 - 2 * c))(np.clip(x, 0, 1))
    Xm = 0.5 * O[:, 0].max()
    sig = sm((O[:, 0] - Xm) / 0.9 + 0.5)
    a = sm(t / 0.45)
    psi = np.pi * a * sig
    E = np.c_[O[:, 0] * (1 - 0.35 * a), O[:, 1] * np.cos(psi), O[:, 1] * np.sin(psi)]
    b = sm((t - 0.45) / 0.55)
    return (1 - b)[..., None] * E + b * C + np.c_[np.zeros(len(O)), np.zeros(len(O)), 0.4 * np.sin(np.pi * b) * sig]


class Flower8:
    def __init__(self, dt=1.2e-4, fabric_gsm=80.0, fabric_Et=3000.0, rod_d=10):
        self.dt = dt
        self.meshes = {False: MC.build(), True: MC.build(blade_len=3.6, width=3.9)}
        offs, Ps, pid = [0], [], []
        for k, pc in enumerate(PETALS):
            m = self.meshes[pc['big']]
            Ps.append(m['P']); pid.append(np.full(len(m['P']), k)); offs.append(offs[-1] + len(m['P']))
        self.offs = np.array(offs)
        P = np.concatenate(Ps); self.P = P; N = len(P); self.N = N
        self.pid_np = np.concatenate(pid); self.pid = jnp.asarray(self.pid_np)
        sp = P[:, 0] - MC.SH
        self.sp, self.yy = jnp.asarray(sp), jnp.asarray(P[:, 1])
        fixed = sp <= 1e-6
        rod = RM = rod_mat(rod_d)
        self.rod = rod
        rigid = np.zeros(N, bool)
        chains, E, L0, K, tris = [], [], [], [], []
        m_true = np.zeros(N)
        self.targets_O, self.targets_C, self.coilR, self.loopL = [], [], [], []
        for k, pc in enumerate(PETALS):
            m, o = self.meshes[pc['big']], offs[k]
            ch = m['perimeter'] + o
            chains.append(ch)
            rigid[ch[(sp[ch] < 0.18) & (sp[ch] > 1e-6)]] = True
            e = m['edges']; l0 = np.hypot(*(m['P'][e[:, 0]] - m['P'][e[:, 1]]).T)
            E.append(e + o); L0.append(l0); K.append(np.minimum(fabric_Et * m['edge_area'] / l0 ** 2, 1.4 * fabric_Et)); tris.append(m['tri'] + o)
            m_true[o:o + len(m['P'])] += fabric_gsm / 1e3 * m['node_area']
            seg = np.hypot(*np.diff(P[ch], axis=0).T)
            mr = np.zeros(len(ch)); mr[:-1] += seg / 2; mr[1:] += seg / 2
            np.add.at(m_true, ch, rod['lam'] * mr)
            O, Cc, R, Lp = coil_targets(m, m['perimeter'])
            self.targets_O.append(O); self.targets_C.append(Cc); self.coilR.append(R); self.loopL.append(Lp)
        self.fixed_np, self.rigid_np = fixed, rigid & ~fixed
        self.free_np = ~(fixed | rigid)
        self.fixed, self.rigid, self.free = (jnp.asarray(a)[:, None] for a in (fixed, self.rigid_np, self.free_np))
        E = np.concatenate(E)
        self.e0, self.e1 = jnp.asarray(E[:, 0]), jnp.asarray(E[:, 1])
        self.eL0, self.ek = jnp.asarray(np.concatenate(L0)), jnp.asarray(np.concatenate(K))
        self.tri_np = np.concatenate(tris); self.tri_pid = self.pid_np[self.tri_np[:, 0]]
        self.chains = [jnp.asarray(c) for c in chains]
        self.chain_rl = [jnp.asarray(np.hypot(*np.diff(P[c], axis=0).T)) for c in chains]
        self.ch_all = np.concatenate(chains); self.ch_all_j = jnp.asarray(self.ch_all)
        self.ch_pid = jnp.asarray(self.pid_np[self.ch_all])
        self.ch_free = jnp.asarray(self.free_np[self.ch_all])[:, None]
        self.m_true_np = m_true; self.m_true = jnp.asarray(m_true)
        self.m = jnp.asarray(np.maximum(m_true, 3e-3))[:, None]
        az = np.deg2rad([pc['az'] for pc in PETALS]); self.az = az
        self.cosA, self.sinA = jnp.asarray(np.cos(az)), jnp.asarray(np.sin(az))
        self.center = jnp.asarray(np.c_[RING_R * np.cos(az), RING_R * np.sin(az), np.full(5, HINGE_Z)])
        # tensores: de la punta del mástil (centro) a la punta y a los dos costados de cada pétalo
        cn = []
        for k, pc in enumerate(PETALS):
            cn += list(self.meshes[pc['big']]['cinch'] + offs[k])
        self.c_idx = jnp.asarray(cn); self.c_pid = jnp.asarray(np.repeat(np.arange(5), 3))
        self.mast = jnp.asarray([-RING_R, 0.0, MAST_H])
        self.c_k = 20000.0
        th_open = np.deg2rad([pc['th_open'] for pc in PETALS])
        flat = np.c_[P[:, 0] - MC.SH, P[:, 1], np.zeros(N)]
        self.cable_L = []
        for j, n in enumerate(cn):
            t = th_open[j // 3]
            pnt = np.array([flat[n, 0] * np.cos(t), flat[n, 1], flat[n, 0] * np.sin(t)])
            self.cable_L.append(np.linalg.norm(pnt - np.asarray(self.mast)))
        self.cable_L = jnp.asarray(self.cable_L)
        self.k_t = 300.0
        self.I_th, self.Kp, self.Kd, self.Mmax = 0.08, 300.0, 30.0, 100.0
        self._grad = jax.grad(self.energy, argnums=(0, 1))
        self._chunk = jax.jit(self._run_chunk, static_argnums=(4,))

    def kin(self, th):
        t = th[self.pid]; c, s = jnp.cos(t), jnp.sin(t)
        rig = jnp.stack([self.sp * c, self.yy, self.sp * s], 1)
        fix = jnp.stack([self.sp, self.yy, jnp.zeros_like(self.sp)], 1)
        return jnp.where(self.fixed, fix, rig)

    def eff(self, x, th):
        return jnp.where(self.free, x, self.kin(th))

    def to_global(self, x):
        ca, sa = self.cosA[self.pid], self.sinA[self.pid]
        return jnp.stack([ca * x[:, 0] - sa * x[:, 1], sa * x[:, 0] + ca * x[:, 1], x[:, 2]], 1) + self.center[self.pid]

    def cable_stretch(self, x, slack):
        d = x[self.c_idx] - self.mast[None, :]
        dist = jnp.sqrt(jnp.sum(d * d, 1) + EPS)
        return jnp.maximum(0.0, dist - self.cable_L * (1 + slack[self.c_pid]))

    def energy(self, x, th, slack, tgt, w):
        x = self.eff(x, th)
        d = x[self.e1] - x[self.e0]
        L = jnp.sqrt(jnp.sum(d * d, 1) + EPS); dl = L - self.eL0
        E = 0.5 * jnp.sum(jnp.where(dl > 0, 1.0, 0.02) * self.ek * dl * dl)
        for idx, rl in zip(self.chains, self.chain_rl):
            r = x[idx]; e = r[1:] - r[:-1]
            le = jnp.sqrt(jnp.sum(e * e, 1) + EPS); tt = e / le[:, None]; dT = tt[1:] - tt[:-1]
            E += 0.5 * self.rod['EI'] * jnp.sum(jnp.sum(dT * dT, 1) / (0.5 * (rl[:-1] + rl[1:])))
            E += 0.5 * (2.0e4 / rl.mean()) * jnp.sum((le - rl) ** 2)
        E += G * jnp.sum(self.m_true * x[:, 2])
        E += 0.5 * self.c_k * jnp.sum(self.cable_stretch(x, slack) ** 2)
        # guía del plegado (las manos del técnico / la funda que dosifica el despliegue)
        dg = x[self.ch_all_j] - tgt
        E += 0.5 * self.k_t * jnp.sum(w[self.ch_pid][:, None] * self.ch_free * dg * dg)
        return E

    def _step(self, state, ctrl, tgt):
        x, v, th, om = state
        th_t, slack, w, damp = ctrl[:5], ctrl[5:10], ctrl[10:15], ctrl[15]
        gx, gth = self._grad(x, th, slack, tgt, w)
        v = (v - self.dt * gx / self.m) * jnp.exp(-damp * self.dt)
        spd = jnp.sqrt(jnp.sum(v * v, 1, keepdims=True) + EPS)
        v = jnp.where(self.free, v * jnp.minimum(1.0, 4.0 / spd), 0.0)
        x = x + self.dt * v
        M = jnp.clip(self.Kp * (th_t - th) - self.Kd * om, -self.Mmax, self.Mmax)
        om = (om + self.dt * (-gth + M) / self.I_th) * jnp.exp(-6.0 * self.dt)
        th = th + self.dt * om
        x = jnp.where(self.free, x, self.kin(th))
        T = self.c_k * self.cable_stretch(x, slack)
        return (x, v, th, om), (M, T)

    def _run_chunk(self, state, c0, c1, tgts, n):
        t0, t1 = tgts
        def body(carry, i):
            f = (i + 1.0) / n
            return self._step(carry, c0 + (c1 - c0) * f, t0 + (t1 - t0) * f)
        state, (M, T) = jax.lax.scan(body, state, jnp.arange(n))
        return state, M.mean(0), T.mean(0)

    def targets(self, tfold, th):
        """Posiciones guía de la varilla (en el marco local de cada pétalo) para el avance de plegado tfold[k] y el ángulo th[k]."""
        out = []
        for k in range(5):
            Q = eight_path(self.targets_O[k], self.targets_C[k], tfold[k])
            c, s = np.cos(th[k]), np.sin(th[k])
            out.append(np.c_[Q[:, 0] * c - Q[:, 2] * s, Q[:, 1], Q[:, 0] * s + Q[:, 2] * c])
        return jnp.asarray(np.concatenate(out))

    def init_state(self, th):
        th = jnp.asarray(th, float)
        return (self.kin(th), jnp.zeros((self.N, 3)), th, jnp.zeros(5))
