"""Opción C: cinco pétalos simulados juntos, con contacto entre todos ellos (y de cada uno consigo mismo).
Cada pétalo: uña fija desde el centro hasta el aro, bisagra en el aro, lámina con brazo rígido, travesaño,
antena y flex de contorno (varilla de fibra de vidrio), cordón de contención de tres ramales.
Coordenadas locales por pétalo: x radial hacia fuera desde la bisagra, y tangente, z arriba."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree
import mesh_c as MC

jax.config.update('jax_enable_x64', True)
G = 9.81
EPS = 1e-12
GF_E = 40e9


def rod_mat(d_mm):
    d = d_mm / 1e3
    return dict(EI=GF_E * np.pi * d ** 4 / 64, lam=1900 * np.pi * d ** 2 / 4)


ALU25_LAM = 2700 * np.pi * (0.025 ** 2 - 0.022 ** 2) / 4

# disposición de orquídea: el lila grande al frente, naranjas a los lados, dos lilas atrás
PETALS = [
    dict(name='Lila grande (labelo)', az=-90, big=True, tex=2, order=0, th_open=-2.0),
    dict(name='Naranja derecho', az=-18, big=False, tex=1, order=1, th_open=6.0),
    dict(name='Lila derecho', az=54, big=False, tex=3, order=3, th_open=16.0),
    dict(name='Lila izquierdo', az=126, big=False, tex=0, order=4, th_open=22.0),
    dict(name='Naranja izquierdo', az=-162, big=False, tex=4, order=2, th_open=10.0),
]
RING_R, HINGE_Z = 1.0, 0.15


class Flower:
    def __init__(self, dt=1.5e-4, fabric_gsm=80.0, fabric_Et=3000.0, thick=0.025, rod_d=6, spreader_d=8, pre=0.01):
        self.dt, self.thick = dt, thick
        self.meshes = {False: MC.build(), True: MC.build(blade_len=3.6, width=3.9)}
        offs, Ps, pid = [0], [], []
        for k, pc in enumerate(PETALS):
            m = self.meshes[pc['big']]
            Ps.append(m['P']); pid.append(np.full(len(m['P']), k)); offs.append(offs[-1] + len(m['P']))
        self.offs = np.array(offs)
        P = np.concatenate(Ps); self.P = P; N = len(P); self.N = N
        self.pid_np = np.concatenate(pid); self.pid = jnp.asarray(self.pid_np)
        sp = P[:, 0] - MC.SH
        self.sp = jnp.asarray(sp); self.yy = jnp.asarray(P[:, 1])
        fixed = np.concatenate([self.meshes[pc['big']]['fixed'] for pc in PETALS])
        rigid = np.concatenate([self.meshes[pc['big']]['rigid'] for pc in PETALS])
        self.fixed_np, self.rigid_np = fixed, rigid
        self.free_np = ~(fixed | rigid)
        self.fixed, self.rigid = jnp.asarray(fixed)[:, None], jnp.asarray(rigid)[:, None]
        self.free = jnp.asarray(self.free_np)[:, None]
        az = np.deg2rad([pc['az'] for pc in PETALS])
        self.az = az
        self.cosA, self.sinA = jnp.asarray(np.cos(az)), jnp.asarray(np.sin(az))
        self.center = jnp.asarray(np.c_[RING_R * np.cos(az), RING_R * np.sin(az), np.full(5, HINGE_Z)])
        # tela
        E, L0, K, tris = [], [], [], []
        m_true = np.zeros(N)
        for k, pc in enumerate(PETALS):
            m, o = self.meshes[pc['big']], offs[k]
            e = m['edges']
            l0 = np.hypot(*(m['P'][e[:, 0]] - m['P'][e[:, 1]]).T)
            E.append(e + o); L0.append(l0); K.append(np.minimum(fabric_Et * m['edge_area'] / l0 ** 2, 1.4 * fabric_Et))
            tris.append(m['tri'] + o)
            m_true[o:o + len(m['P'])] += fabric_gsm / 1e3 * m['node_area']
        E = np.concatenate(E)
        self.e0, self.e1 = jnp.asarray(E[:, 0]), jnp.asarray(E[:, 1])
        self.eL0, self.ek = jnp.asarray(np.concatenate(L0)), jnp.asarray(np.concatenate(K))
        self.tri_np = np.concatenate(tris); self.tri = jnp.asarray(self.tri_np)
        Et = np.sort(np.r_[self.tri_np[:, [0, 1]], self.tri_np[:, [1, 2]], self.tri_np[:, [2, 0]]], 1)
        Eu, cnt = np.unique(Et, axis=0, return_counts=True)
        bd = Eu[cnt == 1]
        self.ma_np, self.mb_np = bd[:, 0], bd[:, 1]
        self.ma, self.mb = jnp.asarray(bd[:, 0]), jnp.asarray(bd[:, 1])
        self.NX = N + len(bd)
        self.pidX_np = np.r_[self.pid_np, self.pid_np[bd[:, 0]]]
        self.PX = np.r_[P, 0.5 * (P[bd[:, 0]] + P[bd[:, 1]])]
        self.tri_pid = self.pid_np[self.tri_np[:, 0]]
        # varillas
        R = rod_mat(rod_d); S = rod_mat(spreader_d)
        self.rods = []
        self.rod_nodes = {'perimeter': [], 'antenna': [], 'spreader': []}
        for k, pc in enumerate(PETALS):
            m, o = self.meshes[pc['big']], offs[k]
            for key, mat, pr in (('perimeter', R, pre), ('antenna', R, 0.0), ('spreader', S, 0.0)):
                idx = m[key] + o
                rl = np.hypot(*(P[idx[1:]] - P[idx[:-1]]).T) * (1 + pr)
                self.rods.append((jnp.asarray(idx), jnp.asarray(rl), mat['EI'], 2.0e4 / rl.mean()))
                seg = rl / (1 + pr)
                mr = np.zeros(len(idx)); mr[:-1] += seg / 2; mr[1:] += seg / 2
                np.add.at(m_true, idx, mat['lam'] * mr)
                self.rod_nodes[key].append(m[key])
            arm = m['arm'] + o
            seg = np.diff(P[arm, 0])
            np.add.at(m_true, arm[:-1], ALU25_LAM * seg / 2); np.add.at(m_true, arm[1:], ALU25_LAM * seg / 2)
        self.m_true_np = m_true
        self.m_true = jnp.asarray(m_true)
        self.m = jnp.asarray(np.maximum(m_true, 1.0e-3))[:, None]
        # cordón de contención: punta → costado del brazo; lados → cruce de brazo y travesaño
        cn, ca, cL0 = [], [], []
        for k, pc in enumerate(PETALS):
            m, o = self.meshes[pc['big']], offs[k]
            for j, n in enumerate(m['cinch']):
                a = (1.0, 0.9) if j == 0 else (1.0, 0.0)
                cn.append(n + o); ca.append(a)
                cL0.append(np.hypot(P[n + o, 0] - MC.SH - a[0], P[n + o, 1] - a[1]) + 0.03)
        self.c_idx = jnp.asarray(cn); self.c_anchor = jnp.asarray(ca); self.c_L0 = jnp.asarray(cL0)
        self.c_pid = jnp.asarray(np.repeat(np.arange(5), 3)); self.c_Lmin = 0.6; self.c_k = 1500.0
        self.cinch_nodes = [m_['cinch'] for m_ in [self.meshes[pc['big']] for pc in PETALS]]
        self.I_th, self.Kp, self.Kd, self.Mmax = 0.08, 300.0, 30.0, 100.0
        self.k_c, self.c_c, self.ct = 1500.0, 1.0, 0.3
        self.pairs = None
        self._grad = jax.grad(self.energy, argnums=(0, 1))
        self._chunk = jax.jit(self._run_chunk, static_argnums=(4,))

    # ---------- geometría
    def kin(self, th):
        t = th[self.pid]
        c, s = jnp.cos(t), jnp.sin(t)
        rig = jnp.stack([self.sp * c, self.yy, self.sp * s], axis=1)
        fix = jnp.stack([self.sp, self.yy, jnp.zeros_like(self.sp)], axis=1)
        return jnp.where(self.fixed, fix, rig)

    def eff(self, x, th):
        return jnp.where(self.free, x, self.kin(th))

    def to_global(self, x):
        ca, sa = self.cosA[self.pid], self.sinA[self.pid]
        gx = ca * x[:, 0] - sa * x[:, 1]
        gy = sa * x[:, 0] + ca * x[:, 1]
        return jnp.stack([gx, gy, x[:, 2]], axis=1) + self.center[self.pid]

    def to_local_vec(self, F):
        ca, sa = self.cosA[self.pid], self.sinA[self.pid]
        return jnp.stack([ca * F[:, 0] + sa * F[:, 1], -sa * F[:, 0] + ca * F[:, 1], F[:, 2]], axis=1)

    def cinch_stretch(self, x, th, cz):
        t = th[self.c_pid]
        nA = jnp.stack([-jnp.sin(t), jnp.zeros_like(t), jnp.cos(t)], axis=1)
        a = self.c_anchor
        anchor = jnp.stack([a[:, 0] * jnp.cos(t), a[:, 1], a[:, 0] * jnp.sin(t)], axis=1) - 0.06 * nA
        d = x[self.c_idx] - anchor
        dist = jnp.sqrt(jnp.sum(d * d, axis=1) + EPS)
        L = self.c_Lmin + (self.c_L0 - self.c_Lmin) * cz[self.c_pid]
        return jnp.maximum(0.0, dist - L)

    def energy(self, x, th, cz):
        x = self.eff(x, th)
        d = x[self.e1] - x[self.e0]
        L = jnp.sqrt(jnp.sum(d * d, axis=1) + EPS)
        dl = L - self.eL0
        E = 0.5 * jnp.sum(jnp.where(dl > 0, 1.0, 0.02) * self.ek * dl * dl)
        for idx, rl, EI, ks in self.rods:
            r = x[idx]
            e = r[1:] - r[:-1]
            le = jnp.sqrt(jnp.sum(e * e, axis=1) + EPS)
            tt = e / le[:, None]
            dT = tt[1:] - tt[:-1]
            E += 0.5 * EI * jnp.sum(jnp.sum(dT * dT, axis=1) / (0.5 * (rl[:-1] + rl[1:])))
            E += 0.5 * ks * jnp.sum((le - rl) ** 2)
        E += G * jnp.sum(self.m_true * x[:, 2])
        E += 0.5 * self.c_k * jnp.sum(self.cinch_stretch(x, th, cz) ** 2)
        g = self.to_global(x)
        on_deck = g[:, 0] ** 2 + g[:, 1] ** 2 < 1.05 ** 2
        pen = jnp.where(on_deck, jnp.maximum(0.0, 0.0 - g[:, 2]), 0.0)
        E += 0.5 * 3000.0 * jnp.sum(pen ** 2)
        return E

    # ---------- contacto vértice–triángulo
    def contact(self, x, v, pairs):
        pv, pt, sg, act = pairs
        g = self.to_global(x)
        ca, sa = self.cosA[self.pid], self.sinA[self.pid]
        vgl = jnp.stack([ca * v[:, 0] - sa * v[:, 1], sa * v[:, 0] + ca * v[:, 1], v[:, 2]], axis=1)
        gX = jnp.concatenate([g, 0.5 * (g[self.ma] + g[self.mb])])
        vX = jnp.concatenate([vgl, 0.5 * (vgl[self.ma] + vgl[self.mb])])
        tri = self.tri[pt]
        a, b, c = g[tri[:, 0]], g[tri[:, 1]], g[tri[:, 2]]
        p = gX[pv]
        n = jnp.cross(b - a, c - a)
        n = n / jnp.sqrt(jnp.sum(n * n, axis=1, keepdims=True) + EPS)
        dist = jnp.sum((p - a) * n, axis=1)
        q = p - dist[:, None] * n
        v0, v1, v2 = b - a, c - a, q - a
        d00, d01, d11 = jnp.sum(v0 * v0, 1), jnp.sum(v0 * v1, 1), jnp.sum(v1 * v1, 1)
        d20, d21 = jnp.sum(v2 * v0, 1), jnp.sum(v2 * v1, 1)
        den = d00 * d11 - d01 * d01 + EPS
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        w0 = 1 - w1 - w2
        inside = (w0 > -0.25) & (w1 > -0.25) & (w2 > -0.25)
        pen = jnp.clip(self.thick - sg * dist, 0.0, 0.08)
        on = act & inside & (pen > 0)
        W = jnp.clip(jnp.stack([w0, w1, w2], 1), 0.0, 1.0)
        W = W / (jnp.sum(W, 1, keepdims=True) + EPS)
        vt = jnp.sum(vgl[tri] * W[:, :, None], axis=1)
        rel = vX[pv] - vt
        vn = sg * jnp.sum(rel * n, axis=1)
        fn = jnp.where(on, jnp.maximum(0.0, self.k_c * pen - self.c_c * vn), 0.0)
        rel_t = rel - jnp.sum(rel * n, axis=1, keepdims=True) * n
        ft = jnp.where(on[:, None], -self.ct * rel_t, 0.0)
        ftn = jnp.sqrt(jnp.sum(ft * ft, 1, keepdims=True) + EPS)
        ft = ft * jnp.minimum(1.0, 0.5 * fn[:, None] / ftn)
        f = (sg * fn)[:, None] * n + ft
        FX = jnp.zeros((self.NX, 3)).at[pv].add(f)
        F = FX[:self.N].at[self.ma].add(0.5 * FX[self.N:]).at[self.mb].add(0.5 * FX[self.N:])
        F = F.at[tri.reshape(-1)].add((-f[:, None, :] * W[:, :, None]).reshape(-1, 3))
        return self.to_local_vec(F), jnp.sum(on)

    def _step(self, state, ctrl, pairs):
        x, v, th, om = state
        th_t, cz, damp = ctrl[:5], ctrl[5:10], ctrl[10]
        gx, gth = self._grad(x, th, cz)
        Fc, n_on = self.contact(self.eff(x, th), v, pairs)
        F = -gx + Fc
        v = (v + self.dt * F / self.m) * jnp.exp(-damp * self.dt)
        sp_ = jnp.sqrt(jnp.sum(v * v, 1, keepdims=True) + EPS)
        v = v * jnp.minimum(1.0, 3.0 / sp_)          # resguardo numérico: ningún nodo a más de 3 m/s
        v = jnp.where(self.free, v, 0.0)
        x = x + self.dt * v
        # bisagras: torque del contacto sobre los nodos rígidos
        t = th[self.pid]
        dk = jnp.stack([-self.sp * jnp.sin(t), jnp.zeros_like(t), self.sp * jnp.cos(t)], 1)
        tau_c = jax.ops.segment_sum(jnp.where(self.rigid[:, 0], jnp.sum(Fc * dk, 1), 0.0), self.pid, 5)
        M = jnp.clip(self.Kp * (th_t - th) - self.Kd * om, -self.Mmax, self.Mmax)
        om = (om + self.dt * (-gth + tau_c + M) / self.I_th) * jnp.exp(-6.0 * self.dt)
        th = th + self.dt * om
        x = jnp.where(self.free, x, self.kin(th))
        return (x, v, th, om), (M, -gth, tau_c, n_on)

    def _run_chunk(self, state, c0, c1, pairs, n):
        def body(carry, i):
            f = (i + 1.0) / n
            return self._step(carry, c0 + (c1 - c0) * f, pairs)
        state, (M, Qg, tc, non) = jax.lax.scan(body, state, jnp.arange(n))
        return state, M.mean(0), Qg.mean(0), tc.mean(0), non.max()

    # ---------- lista de pares (fuera de jit)
    def build_pairs(self, x, th, r=0.35, kq=12):
        xe = np.asarray(self.eff(jnp.asarray(x), jnp.asarray(th)))
        g = np.asarray(self.to_global(jnp.asarray(xe)))
        gX = np.r_[g, 0.5 * (g[self.ma_np] + g[self.mb_np])]
        cen = g[self.tri_np].mean(1)
        cenP = self.P[self.tri_np].mean(1)
        PV, PT = [], []
        for k in range(5):
            ti = np.where(self.tri_pid == k)[0]
            tree = cKDTree(cen[ti])
            d, j = tree.query(gX, k=kq, distance_upper_bound=r)
            ok = np.isfinite(d)
            vi = np.repeat(np.arange(self.NX)[:, None], kq, 1)[ok]
            tj = ti[np.minimum(j[ok], len(ti) - 1)]
            same = self.pidX_np[vi] == k
            far = np.hypot(*(self.PX[vi] - cenP[tj]).T) > 0.4
            keep = ~same | far
            vi, tj = vi[keep], tj[keep]
            isN = vi < self.N
            ea = np.where(isN, vi, self.ma_np[np.maximum(vi - self.N, 0)]); eb = np.where(isN, vi, self.mb_np[np.maximum(vi - self.N, 0)])
            notin = ~np.any((self.tri_np[tj] == ea[:, None]) | (self.tri_np[tj] == eb[:, None]), axis=1)
            vnode = ea
            stat_v = ~self.free_np[ea] & ~self.free_np[eb]
            stat_t = np.all(~self.free_np[self.tri_np[tj]], axis=1)
            same_p_t = np.all(self.pid_np[self.tri_np[tj]] == self.pidX_np[vi][:, None], axis=1)
            both_static = stat_v & stat_t & (same_p_t | (self.fixed_np[ea] & self.fixed_np[eb] & np.all(self.fixed_np[self.tri_np[tj]], axis=1)))
            keep = notin & ~both_static
            PV.append(vi[keep]); PT.append(tj[keep])
        pv, pt = np.concatenate(PV), np.concatenate(PT)
        a, b, c = g[self.tri_np[pt, 0]], g[self.tri_np[pt, 1]], g[self.tri_np[pt, 2]]
        n = np.cross(b - a, c - a); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        dist = np.einsum('ij,ij->i', gX[pv] - a, n)
        sg = np.where(dist >= 0, 1.0, -1.0)
        key = pv.astype(np.int64) * len(self.tri_np) + pt
        if self.pairs is not None and len(self.pairs_key):  # conservar el lado conocido de cada par
            okey, osg = self.pairs_key, self.pairs_sg
            pos = np.searchsorted(okey, key)
            pos = np.minimum(pos, len(okey) - 1)
            hit = okey[pos] == key
            sg[hit] = osg[pos[hit]]
        order = np.argsort(key)
        self.pairs_key, self.pairs_sg = key[order], sg[order]
        K = len(pv)
        Kp = int(np.ceil(max(K, 1) / 16384) * 16384)
        pad = Kp - K
        self.pairs = (jnp.asarray(np.r_[pv, np.zeros(pad, int)]), jnp.asarray(np.r_[pt, np.zeros(pad, int)]),
                      jnp.asarray(np.r_[sg, np.ones(pad)]), jnp.asarray(np.r_[np.ones(K, bool), np.zeros(pad, bool)]))
        return K

    def init_state(self, th):
        th = jnp.asarray(th, float)
        return (self.kin(th), jnp.zeros((self.N, 3)), th, jnp.zeros(5))
