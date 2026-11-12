import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const D = JSON.parse(document.getElementById('sim-data').textContent);
const $ = id => document.getElementById(id);
const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer;
const DEG = Math.PI / 180;

// ---------- datos
const MESH = D.meshes;
Object.values(MESH).forEach(m => {
  m.P = new Float32Array(b64(m.P));
  m.tri = new Uint16Array(b64(m.tri));
  m.uv = new Float32Array(m.N * 2);
  for (let i = 0; i < m.N; i++) { m.uv[2 * i] = m.P[2 * i + 1] / m.width + 0.5; m.uv[2 * i + 1] = m.P[2 * i] / m.length; }
});
const NH = MESH.A.nHoop;
const maxLen = k => Math.max(1, ...Object.values(MESH).map(m => (m.ribs[k] || []).length));
const ANIMS = D.anims;
Object.values(ANIMS).forEach(a => { a.frames = new Int16Array(b64(a.frames)); });
const isC = m => m === 'florc' || m === 'florcr';
const isAnim = m => m === 'cruceta' || m === 'flex';
const animOf = m => (m === 'flex' ? ANIMS.flex : isC(m) ? ANIMS.florc : ANIMS.cruceta);
D.poses.forEach(p => { p.x = new Int16Array(b64(p.x)); });
const G = D.geom;
const CPOSE = new Int16Array(b64(D.C.x));            // geometría del bastidor y del mecanismo
const TH_OPEN = ANIMS.cruceta.thOpen;

// ---------- escena
const wrap = $('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
wrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
scene.add(new THREE.HemisphereLight(0xffffff, 0x4a5a66, 1.25));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(6, -8, 30);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xfff0dd, 0.9);
fill.position.set(-10, 12, -6);
scene.add(fill);

const mat = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.15, ...o });
const M = {
  frame: mat(0x8fa3ad, { metalness: 0.5, roughness: 0.4 }),
  platform: mat(0xf2b01e),
  scissor: mat(0x5d6f78, { metalness: 0.4 }),
  stage: mat(0x5f8f8c, { transparent: true, opacity: 0.55, roughness: 0.9, side: THREE.DoubleSide }),
  ground: mat(0x22343b, { roughness: 1 }),
  hoop: mat(0x16d4c4, { metalness: 0.6, roughness: 0.3 }),
  arm: mat(0xdfe6ea, { metalness: 0.8, roughness: 0.3 }),
  spreader: mat(0xff8a1f),
  antenna: mat(0xffe23d),
  perimeter: mat(0x9dff2e, { roughness: 0.4 }),
  cinch: mat(0x31e6ff, { roughness: 0.9 }),
  rope: mat(0xffffff, { roughness: 0.9 }),
  bungee: mat(0xff2bd0, { roughness: 0.8 }),
  hardware: mat(0x2f6fe0, { metalness: 0.3 }),
  button: mat(0x7a1750, { roughness: 0.8 }),
};

// tubos por tramos (cilindros instanciados entre puntos consecutivos)
const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 8, 1, true);
unitCyl.rotateX(Math.PI / 2);                       // eje del cilindro → z local
class PolyTube {
  constructor(maxSeg, radius, material, parent) {
    this.mesh = new THREE.InstancedMesh(unitCyl, material, maxSeg);
    this.mesh.frustumCulled = false;
    this.r = radius;
    this.o = new THREE.Object3D();
    this.a = new THREE.Vector3();
    this.b = new THREE.Vector3();
    parent.add(this.mesh);
  }
  set(points, closed = false) {            // points: [[x,y,z], ...]
    const n = points.length;
    const segs = closed ? n : n - 1;
    for (let i = 0; i < segs; i++) {
      const p = points[i], q = points[(i + 1) % n];
      this.a.set(p[0], p[1], p[2]);
      this.b.set(q[0], q[1], q[2]);
      const len = this.a.distanceTo(this.b);
      this.o.position.copy(this.a).add(this.b).multiplyScalar(0.5);
      this.o.lookAt(this.b);
      this.o.scale.set(this.r, this.r, Math.max(len, 1e-5));
      this.o.updateMatrix();
      this.mesh.setMatrixAt(i, this.o.matrix);
    }
    this.mesh.count = Math.max(segs, 0);
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
function bar(a, b, r, material, parent) {
  const t = new PolyTube(1, r, material, parent);
  t.set([a, b]);
  return t;
}

// ---------- contexto: suelo, tarima con boca, tijera
const ground = new THREE.Mesh(new THREE.CircleGeometry(16, 64), M.ground);
scene.add(ground);
const stage = new THREE.Group();
scene.add(stage);
function buildStage(trap) {
  while (stage.children.length) { const c = stage.children.pop(); c.geometry.dispose(); }
  const S = G.stage, h = trap / 2, w = S.size / 2, th = 0.12;
  [[-(w + h) / 2, 0, w - h, S.size], [(w + h) / 2, 0, w - h, S.size], [0, -(w + h) / 2, trap, w - h], [0, (w + h) / 2, trap, w - h]].forEach(([x, y, sx, sy]) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, th), M.stage);
    m.position.set(x, y, S.z - th / 2);
    stage.add(m);
  });
}
const base = new THREE.Mesh(new THREE.BoxGeometry(2.49, 1.2, 0.55), M.platform);
base.position.set(0, 0, 0.42);
scene.add(base);
const scissorBars = [];
for (let i = 0; i < 20; i++) scissorBars.push(new PolyTube(1, 0.035, M.scissor, scene));
function setScissor(H) {
  const z0 = 0.7, stages = 5, hs = (H - 0.12 - z0) / stages, L = 2.25;
  const dx = 0.5 * Math.sqrt(Math.max(L * L - hs * hs, 0.01));
  let k = 0;
  for (let s = 0; s < stages; s++) {
    const za = z0 + s * hs, zb = za + hs;
    for (const y of [-0.5, 0.5]) {
      scissorBars[k++].set([[-dx, y, za], [dx, y, zb]]);
      scissorBars[k++].set([[dx, y, za], [-dx, y, zb]]);
    }
  }
}

// ---------- conjunto que sube con la plataforma
const top = new THREE.Group();
scene.add(top);
{
  const slab = new THREE.Mesh(new THREE.BoxGeometry(G.platform[0], G.platform[1], 0.12), M.platform);
  slab.position.z = -0.06;
  top.add(slab);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(G.ringR, 0.014, 8, 72), M.frame);
  ring.position.z = G.lowerZ;
  top.add(ring);
  const ring2 = new THREE.Mesh(new THREE.TorusGeometry(G.ringR, 0.014, 8, 72), M.frame);
  ring2.position.z = G.z0 - 0.03;
  top.add(ring2);
  for (const x of [-0.75, 0, 0.75]) {
    const y = Math.sqrt(G.ringR ** 2 - x * x);
    bar([x, -y, G.lowerZ], [x, y, G.lowerZ], 0.014, M.frame, top);
  }
  bar([-G.ringR, 0, G.lowerZ], [G.ringR, 0, G.lowerZ], 0.014, M.frame, top);
  bar([0, 0, 0], [0, 0, G.z0 - 0.07], 0.035, M.frame, top);
  const btn = new THREE.Mesh(new THREE.SphereGeometry(0.24, 32, 12, 0, Math.PI * 2, 0, Math.PI / 2), M.button);
  btn.rotation.x = Math.PI / 2;
  btn.scale.set(1, 0.14, 1);
  btn.position.z = G.z0 - 0.075;
  top.add(btn);
}

// ---------- pétalos
const loader = new THREE.TextureLoader();
const textures = D.textures.map(src => { const t = loader.load(src); t.colorSpace = THREE.SRGBColorSpace; return t; });
const petals = [];
const clothMats = [];
for (let i = 0; i < 5; i++) {
  const g = new THREE.Group();
  const az = (90 + i * 144) * DEG;
  g.position.set(G.ringR * Math.cos(az), G.ringR * Math.sin(az), G.z0 + i * G.pitch);
  g.rotation.z = az;
  top.add(g);
  const cm = new THREE.MeshStandardMaterial({ map: textures[D.textureOrder[i]], side: THREE.DoubleSide, roughness: 0.85, metalness: 0, transparent: true, opacity: 1 });
  clothMats.push(cm);
  const sets = {};
  for (const [key, ms] of Object.entries(MESH)) {
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(ms.N * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(ms.uv, 2));
    geo.setIndex(new THREE.BufferAttribute(ms.tri, 1));
    const cloth = new THREE.Mesh(geo, cm);
    cloth.frustumCulled = false;
    g.add(cloth);
    sets[key] = { geo, pos, cloth };
  }
  // herrajes de raíz
  const post = bar([0, 0, -0.012], [0, 0, G.lowerZ - (G.z0 + i * G.pitch)], 0.012, M.frame, g);
  bar([0, -0.34, 0], [0, 0.34, 0], 0.012, M.frame, g);                                   // eje de bisagra
  bar([0, 0.27, 0], [G.fair[0], 0.27, G.fair[1]], 0.008, M.frame, g);                    // soporte del pasacabos
  bar([0, -0.27, 0], [0, -0.27, -G.anchorA], 0.008, M.frame, g);                         // soporte del anclaje del elástico
  const rot = new THREE.Group();
  g.add(rot);
  const plate = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.44, 0.012), M.hardware);
  plate.position.x = 0.035;
  rot.add(plate);
  const sector = new THREE.Mesh(new THREE.CylinderGeometry(G.sectorR, G.sectorR, 0.02, 40, 1, false, -115 * DEG, 180 * DEG), M.hardware);
  sector.position.y = 0.27;
  rot.add(sector);
  const hornAng = (180 + G.deltaDeg) * DEG;
  const horn = bar([0, -0.27, 0], [G.hornB * Math.cos(hornAng), -0.27, G.hornB * Math.sin(hornAng)], 0.01, M.hardware, rot);
  const pt = {
    i, g, rot, sets, post, horn,
    hoop: new PolyTube(NH, 0.009, M.hoop, g),
    arm: new PolyTube(1, 0.0125, M.arm, g),
    spreader: new PolyTube(maxLen('spreader'), 0.006, M.spreader, g),
    antenna: new PolyTube(maxLen('antenna'), 0.006, M.antenna, g),
    perimeter: new PolyTube(maxLen('perimeter'), 0.007, M.perimeter, g),
    cinch: [0, 1, 2].map(() => new PolyTube(1, 0.004, M.cinch, g)),
    rope: new PolyTube(24, 0.0045, M.rope, g),
    bungee: new PolyTube(2, 0.006, M.bungee, g),
  };
  petals.push(pt);
}

function pts(pos, idx) { return idx.map(k => [pos[3 * k], pos[3 * k + 1], pos[3 * k + 2]]); }
const hoopIdx = Array.from({ length: NH }, (_, k) => k);
const fairBeta = Math.atan2(G.fair[1], G.fair[0]);
const fairD = Math.hypot(G.fair[0], G.fair[1]);
const phiT = fairBeta - Math.acos(G.sectorR / fairD) + 2 * Math.PI;      // punto de tangencia de la rienda

function setPetal(pt, key, src, i0, f, th, mode, cinchOn, stride, base) {
  const ms = MESH[key], N = ms.N, set = pt.sets[key], pos = set.pos, R = ms.ribs;
  for (const [k, s2] of Object.entries(pt.sets)) s2.cloth.visible = k === key;
  const S = stride || N, off = (i0 * S + (base || 0)) * 3, off2 = off + S * 3;
  if (f > 0) for (let k = 0; k < N * 3; k++) pos[k] = (src[off + k] * (1 - f) + src[off2 + k] * f) * 0.001;
  else for (let k = 0; k < N * 3; k++) pos[k] = src[off + k] * 0.001;
  set.geo.attributes.position.needsUpdate = true;
  set.geo.computeVertexNormals();
  pt.hoop.set(pts(pos, hoopIdx), true);
  const ribbed = mode !== 'literal';
  const flex = mode === 'flex' || mode === 'flexmat' || isC(mode);
  pt.arm.mesh.visible = pt.spreader.mesh.visible = ribbed && state.showFrame;
  const hasAnt = R.antenna.length > 0 && (mode === 'antena' || flex);
  pt.antenna.mesh.visible = hasAnt && state.showFrame;
  pt.hoop.mesh.visible = state.showFrame;
  pt.perimeter.mesh.visible = flex && state.showFrame;
  if (flex) pt.perimeter.set(pts(pos, R.perimeter), true);
  const nAx = -Math.sin(th), nAz = Math.cos(th);
  pt.cinch.forEach((c, k) => {
    c.mesh.visible = !!cinchOn && state.showLines;
    if (cinchOn) {
      const n = R.cinch[k], a = R.cinchAnchors[k];
      c.set([[pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]], [a[0] * Math.cos(th) - 0.06 * nAx, a[1], a[0] * Math.sin(th) - 0.06 * nAz]]);
    }
  });
  if (ribbed) {
    pt.arm.set([[0, 0, 0], [2 * Math.cos(th), 0, 2 * Math.sin(th)]]);
    pt.spreader.set(pts(pos, R.spreader));
  }
  if (hasAnt) pt.antenna.set(pts(pos, R.antenna));
  pt.rot.rotation.y = -th;
  // rienda: pasacabos → tangente → arrollada en el sector hasta su amarre
  const r = G.sectorR, y = 0.27;
  const a0 = th + 29.8 * DEG, a1 = phiT;
  const rope = [[-G.ringR + 0.04, 0, G.lowerZ - pt.g.position.z + 0.02], [G.fair[0], y, G.fair[1]]];
  const na = Math.max(2, Math.ceil((a1 - a0) / (9 * DEG)));
  for (let k = na; k >= 0; k--) { const a = a0 + (a1 - a0) * k / na; rope.push([r * Math.cos(a), y, r * Math.sin(a)]); }
  pt.rope.set(rope);
  const ha = th + (180 + G.deltaDeg) * DEG;
  pt.bungee.set([[G.hornB * Math.cos(ha), -0.27, G.hornB * Math.sin(ha)], [0, -0.27, -G.anchorA]]);
  pt.rope.mesh.visible = pt.bungee.mesh.visible = state.showLines;
}

// ---------- línea de tiempo
const T_LIFT = 8, STAGGER = 4;
let ANIM = ANIMS.cruceta, T_PET = (ANIM.nF - 1) / ANIM.fps, T_TOTAL = T_LIFT + 4 * STAGGER + T_PET;
const H0 = G.liftStowed, H1 = G.liftTop;
const state = { option: 'A', mode: 'cruceta', pose: null, t: 0, playing: false, follow: true, showFrame: true, showLines: true, sel: 4, lastH: H0 };
const smooth = u => u * u * (3 - 2 * u);
const lerpArr = (arr, f) => { const i = Math.min(Math.floor(f), arr.length - 2); const u = Math.min(f - i, 1); return arr[i] * (1 - u) + arr[i + 1] * u; };
function petalTau(i, t) { return t - T_LIFT - (4 - i) * STAGGER; }

function apply() {
  const t = state.t;
  const H = state.mode === 'florcr' ? H1 : isAnim(state.mode) ? H0 + (H1 - H0) * smooth(Math.min(t / T_LIFT, 1)) : H1;
  if (state.follow) { const dz = H - state.lastH; camera.position.z += dz; controls.target.z += dz; }
  state.lastH = H;
  top.position.z = H;
  setScissor(H);
  if (isC(state.mode)) {
    for (const pt of petals) setPetal(pt, D.C.mesh[pt.i], CPOSE, 0, 0, D.C.th[pt.i], state.mode, false, D.C.N, D.C.offs[pt.i]);
    updateReadout(); drawCharts(); return;
  }
  for (const pt of petals) {
    if (isAnim(state.mode)) {
      const tau = Math.max(0, Math.min(petalTau(pt.i, t), T_PET));
      const ff = tau * ANIM.fps, i0 = Math.min(Math.floor(ff), ANIM.nF - 2), f = ff - i0;
      setPetal(pt, ANIM.mesh, ANIM.frames, i0, f, lerpArr(ANIM.th, ff), state.mode, state.mode === 'flex' && i0 < ANIM.nCinch);
    } else {
      setPetal(pt, state.pose.mesh, state.pose.x, 0, 0, TH_OPEN, state.mode, false);
    }
  }
  updateReadout();
  drawCharts();
}

// ---------- lecturas y gráficas
const fmt = (v, d = 0) => v.toLocaleString('es-CO', { minimumFractionDigits: d, maximumFractionDigits: d });
function cFrame(t) {
  if (state.mode === 'florcr') return Math.min(t * ANIM.fps, ANIM.nClose - 1);
  return ANIM.nClose - 1 + Math.max(0, Math.min((t - T_LIFT) * ANIM.fps, ANIM.nOpen - 1));
}
function cRec(t, col) { const k = Math.round(cFrame(t)); return ANIM.rec[col][Math.min(k, ANIM.nF - 1)]; }
function totalTension(t) { let s = 0; for (let i = 0; i < 5; i++) { const tau = petalTau(i, t); if (tau > 0 && tau < T_PET) s += lerpArr(ANIM.T, tau * ANIM.fps); } return s; }
function updateReadout() {
  $('time').textContent = fmt(state.t, 1) + ' / ' + fmt(T_TOTAL, 1) + ' s';
  const anim = isAnim(state.mode);
  $('slider').disabled = $('play').disabled = !anim;
  if (!anim) { $('phase').textContent = 'Postura abierta en equilibrio. Esta variante no tiene secuencia animada.'; return; }
  const t = state.t;
  if (isC(state.mode)) { $('phase').textContent = 'Postura abierta en equilibrio, los cinco pétalos calculados juntos. La recogida no quedó resuelta: ver el panel.'; return; }
  let ph;
  if (t < T_LIFT) ph = 'Elevación con el paquete retenido (tiempo comprimido: la subida real tarda más de un minuto).';
  else if (t < T_TOTAL - 3) {
    const k = Math.min(4, Math.floor((t - T_LIFT) / STAGGER));
    ph = state.mode === 'flex'
      ? 'Apertura: el pétalo ' + (k + 1) + ' de 5 voltea contenido. Casi vertical se suelta el cordón, el flex abre la tela hacia arriba y el pétalo baja abierto.'
      : 'Apertura: se da rienda al pétalo ' + (k + 1) + ' de 5. El de arriba de la pila sale primero.';
  }
  else ph = 'Abierta: cada pétalo descansa en su tope. Falta el pestillo que lo asegure contra el viento.';
  $('phase').textContent = ph;
  const tau = Math.max(0, Math.min(petalTau(state.sel, t), T_PET)), ff = tau * ANIM.fps;
  $('r-th').textContent = fmt(lerpArr(ANIM.th, ff) / DEG, 0) + '°';
  $('r-T').textContent = fmt(lerpArr(ANIM.T, ff), 0) + ' N';
  $('r-Ttot').textContent = fmt(totalTension(t), 0) + ' N';
  if (state.mode === 'flex') $('r-Tc').textContent = fmt(lerpArr(ANIM.Tc, ff), 0) + ' N';
  $('r-H').textContent = fmt(top.position.z, 2) + ' m';
}
function chart(cv, series, xr, yr, marker, xl, yl, ticksX, ticksY) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), W = cv.clientWidth, Hh = cv.clientHeight;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = Hh * dpr; }
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, Hh);
  const L = 44, R = 10, T = 10, B = 34, X = v => L + (v - xr[0]) / (xr[1] - xr[0]) * (W - L - R), Y = v => Hh - B - (v - yr[0]) / (yr[1] - yr[0]) * (Hh - T - B);
  c.font = '11px Arial'; c.fillStyle = '#c9d6d6'; c.strokeStyle = '#3d555d'; c.lineWidth = 1;
  c.textAlign = 'right'; c.textBaseline = 'middle';
  for (const v of ticksY) { c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(W - R, Y(v)); c.stroke(); c.fillText(fmt(v, 0), L - 6, Y(v)); }
  c.textAlign = 'center'; c.textBaseline = 'top';
  for (const v of ticksX) { c.beginPath(); c.moveTo(X(v), Hh - B); c.lineTo(X(v), Hh - B + 4); c.stroke(); c.fillText(fmt(v, 0), X(v), Hh - B + 6); }
  c.fillText(xl, (L + W - R) / 2, Hh - 13);
  if (yr[0] < 0) { c.strokeStyle = '#8aa0a6'; c.beginPath(); c.moveTo(L, Y(0)); c.lineTo(W - R, Y(0)); c.stroke(); }
  for (const s of series) {
    c.strokeStyle = s.color; c.lineWidth = 2.2; c.beginPath();
    s.x.forEach((x, k) => { const px = X(x), py = Y(s.y[k]); if (k) c.lineTo(px, py); else c.moveTo(px, py); });
    c.stroke();
  }
  if (marker != null) { c.strokeStyle = '#ffffff'; c.lineWidth = 1.5; c.setLineDash([4, 3]); c.beginPath(); c.moveTo(X(marker), T); c.lineTo(X(marker), Hh - B); c.stroke(); c.setLineDash([]); }
}
let tAxis, totSeries, oneSeries, thDeg, flipN;
function prepSeries() {
  tAxis = Array.from({ length: 281 }, (_, k) => T_LIFT + k * (T_TOTAL - T_LIFT) / 280);
  totSeries = tAxis.map(totalTension);
  oneSeries = tAxis.map(t => { const tau = petalTau(4, t); return tau > 0 && tau < T_PET ? lerpArr(ANIM.T, tau * ANIM.fps) : 0; });
  flipN = ANIM.nF;
  thDeg = ANIM.th.slice(0, flipN).map(v => v / DEG);
}
const niceMax = v => Math.max(50, Math.ceil(v / 50) * 50);
function drawChartsC() {
  const k0 = state.mode === 'florcr' ? 0 : ANIM.nClose - 1, k1 = state.mode === 'florcr' ? ANIM.nClose - 1 : ANIM.nF - 1;
  const xs = [], series = [0, 1, 2, 3, 4].map(() => []), cr = [];
  for (let k = k0; k <= k1; k++) {
    const tt = state.mode === 'florcr' ? (k - k0) / ANIM.fps : T_LIFT + (k - k0) / ANIM.fps;
    xs.push(tt);
    for (let p = 0; p < 5; p++) series[p].push(Math.abs(ANIM.M[k * 5 + p]));
    cr.push(ANIM.rec[3][k]);
  }
  const tx = []; for (let v = 0; v < T_TOTAL; v += 10) tx.push(v);
  chart($('c-C1'), series.map((y, p) => ({ x: xs, y, color: D.C.colors[p] })), [xs[0], xs[xs.length - 1]], [0, 70], state.t, 'tiempo (s)', 'N·m', tx, [0, 20, 40, 60]);
  const cmax = Math.max(5, ...cr);
  chart($('c-C2'), [{ x: xs, y: cr, color: '#4fe08a' }], [xs[0], xs[xs.length - 1]], [0, cmax], state.t, 'tiempo (s)', '', tx, [0, Math.round(cmax / 2), Math.round(cmax)]);
}
function drawCharts() {
  if (isC(state.mode)) return;
  if (!isAnim(state.mode) || !tAxis) return;
  const ymax = niceMax(Math.max(...totSeries));
  const ty = []; for (let v = 0; v <= ymax; v += ymax > 200 ? 100 : 50) ty.push(v);
  const tx = []; for (let v = 10; v < T_TOTAL; v += 5) tx.push(v);
  chart($('c-T'), [{ x: tAxis, y: oneSeries, color: '#ffd23d' }, { x: tAxis, y: totSeries, color: '#ff4fd8' }], [T_LIFT, T_TOTAL], [0, ymax],
    Math.max(state.t, T_LIFT), 'tiempo (s)', 'N', tx, ty);
  const tau = Math.max(0, Math.min(petalTau(state.sel, state.t), T_PET));
  const lim = state.mode === 'flex' ? 60 : 30;
  chart($('c-M'), [{ x: thDeg, y: ANIM.Qg.slice(0, flipN), color: '#36c5ff' }, { x: thDeg, y: ANIM.Ms.slice(0, flipN), color: '#ff4fd8' }], [180, 0], [-lim, lim],
    lerpArr(ANIM.th, tau * ANIM.fps) / DEG, 'ángulo del pétalo (°): 180 recogido → 12 abierto', 'N·m', [180, 135, 90, 45, 0], [-lim, -lim / 2, 0, lim / 2, lim]);
}

// ---------- vistas
function view(name) {
  const H = top.position.z;
  const V = {
    conjunto: [[13, -15, H * 0.55 + 4], [0, 0, H * 0.55]],
    corola: [[6.5, -7.5, H + 4.2], [0, 0, H + 0.2]],
    publico: [[7, -9, 1.7], [0, 0, H - 0.5]],
    raiz: [[1.25, 2.55, H + 0.95], [0, 0.95, H + 0.12]],
    planta: [[0, -0.9, H + 13], [0, 0, H]],
  }[name];
  camera.position.set(...V[0]);
  controls.target.set(...V[1]);
  controls.update();
}

// ---------- interfaz
function layoutPetals(pitch, C) {
  for (const pt of petals) {
    const az = C ? D.C.az[pt.i] * DEG : (90 + pt.i * 144) * DEG;
    pt.g.position.set(G.ringR * Math.cos(az), G.ringR * Math.sin(az), C ? D.C.hingeZ : G.z0 + pt.i * pitch);
    pt.g.rotation.z = az;
    pt.post.set([[0, 0, -0.012], [0, 0, G.lowerZ - pt.g.position.z]]);
    clothMats[pt.i].map = textures[C ? D.C.tex[pt.i] : D.textureOrder[pt.i]];
    clothMats[pt.i].needsUpdate = true;
  }
}
const OPTION_OF = { literal: 'A', cruceta: 'A', antena: 'A', flex: 'B', flexmat: 'B', florc: 'C', florcr: 'C' };
function setMode(mode, poseId) {
  const wasAnim = isAnim(state.mode), prev = state.mode;
  state.mode = mode;
  state.option = OPTION_OF[mode];
  layoutPetals(state.option === 'B' ? G.pitchB : G.pitch, state.option === 'C');
  buildStage(state.option === 'B' ? G.stage.trapB : state.option === 'C' ? G.stage.trapC : G.stage.trapA);
  document.querySelectorAll('[data-option]').forEach(b => b.setAttribute('aria-pressed', b.dataset.option === state.option));
  document.querySelectorAll('[data-mode]').forEach(b => { b.setAttribute('aria-pressed', b.dataset.mode === mode); b.hidden = OPTION_OF[b.dataset.mode] !== state.option; });
  document.querySelectorAll('[data-show]').forEach(el => { el.hidden = !el.dataset.show.split(' ').some(k => k === mode || k === 'opt' + state.option); });
  if (isC(mode)) { state.playing = false; $('play').textContent = '▶ Reproducir'; apply(); return; }
  if (isAnim(mode)) {
    ANIM = animOf(mode);
    T_PET = isC(mode) ? (mode === 'florc' ? ANIM.nOpen - 1 : ANIM.nClose - 1) / ANIM.fps : (ANIM.nF - 1) / ANIM.fps;
    T_TOTAL = mode === 'florc' ? T_LIFT + T_PET : mode === 'florcr' ? T_PET : T_LIFT + 4 * STAGGER + T_PET;
    $('slider').max = T_TOTAL;
    if (!wasAnim || prev !== mode) state.t = Math.min(state.t, T_TOTAL);
    $('slider').value = state.t;
    if (!isC(mode)) prepSeries();
  } else {
    const cfg = mode === 'flexmat' ? 'flex' : mode;
    const list = D.poses.filter(p => p.config === cfg);
    state.pose = list.find(p => p.id === poseId) || list.find(p => p.default) || list[0];
    const sel = $('pose-select');
    sel.innerHTML = '';
    list.forEach(p => { const o = document.createElement('option'); o.value = p.id; o.textContent = p.label; o.selected = p === state.pose; sel.appendChild(o); });
    $('pose-label').textContent = { literal: 'Material del aro', antena: 'Varilla de la antena', flexmat: 'Material del flex de contorno' }[mode];
    const p = state.pose;
    const cell = (a, b) => `<div><dt>${a}</dt><dd>${b}</dd></div>`;
    const upDown = v => (v >= 0 ? fmt(v, 2) + ' m abajo' : fmt(-v, 2) + ' m arriba');
    $('pose-read').innerHTML = mode === 'literal'
      ? cell('Borde lejano del aro', fmt(p.far_drop, 2) + ' m bajo su plano') + cell('Punto más bajo de la tela', fmt(-p.low_rel, 2) + ' m bajo la raíz') + cell('Alcance horizontal', fmt(p.reach, 2) + ' m de 4,00') + cell('Masa del pétalo', fmt(p.mass, 2) + ' kg')
      : mode === 'antena'
        ? cell('Caída de la punta', fmt(p.tip_drop, 2) + ' m') + cell('Alcance horizontal', fmt(p.reach, 2) + ' m de 4,00') + cell('Esfuerzo en la varilla', fmt(p.antenna_sigma, 0) + ' MPa') + cell('Par en la raíz', fmt(-p.Qg, 0) + ' N·m')
        : cell('Punta respecto al plano', upDown(p.tip_h)) + cell('Lados respecto al plano', upDown(p.side_h)) + cell('Ancho visible', fmt(p.span, 2) + ' m de 3,50') + cell('Masa del pétalo', fmt(p.mass, 1) + ' kg');
    state.playing = false;
    $('play').textContent = '▶ Reproducir';
  }
  apply();
}
document.querySelectorAll('[data-option]').forEach(b => b.addEventListener('click', () => setMode({ A: 'cruceta', B: 'flex', C: 'florc' }[b.dataset.option])));
document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
$('pose-select').addEventListener('change', e => setMode(state.mode, e.target.value));
document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => view(b.dataset.view)));
$('slider').addEventListener('input', e => { state.t = +e.target.value; apply(); });
$('play').addEventListener('click', () => {
  if (state.t >= T_TOTAL - 0.05) state.t = 0;
  state.playing = !state.playing;
  $('play').textContent = state.playing ? '❚❚ Pausa' : '▶ Reproducir';
});
document.querySelectorAll('[data-jump]').forEach(b => b.addEventListener('click', () => {
  state.t = isC(state.mode) ? { guardada: 0, arriba: state.mode === 'florcr' ? T_TOTAL * 0.25 : T_LIFT, mitad: state.mode === 'florcr' ? T_TOTAL * 0.6 : T_LIFT + T_PET * 0.45, abierta: T_TOTAL }[b.dataset.jump] : { guardada: 0, arriba: T_LIFT, mitad: T_LIFT + 2 * STAGGER + T_PET / 2, abierta: T_TOTAL }[b.dataset.jump];
  state.playing = false; $('play').textContent = '▶ Reproducir'; $('slider').value = state.t; apply();
}));
$('opacity').addEventListener('input', e => { const v = +e.target.value; clothMats.forEach(m => { m.opacity = v; m.depthWrite = v > 0.98; }); $('opacity-out').textContent = Math.round(v * 100) + ' %'; });
$('show-frame').addEventListener('change', e => { state.showFrame = e.target.checked; apply(); });
$('show-lines').addEventListener('change', e => { state.showLines = e.target.checked; apply(); });
$('show-stage').addEventListener('change', e => { stage.visible = e.target.checked; });
$('follow').addEventListener('change', e => { state.follow = e.target.checked; });

function resize() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  drawCharts();
}
window.addEventListener('resize', resize);
let last = performance.now();
function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (state.playing) {
    state.t = Math.min(state.t + dt, T_TOTAL);
    if (state.t >= T_TOTAL) { state.playing = false; $('play').textContent = '▶ Reproducir'; }
    $('slider').value = state.t;
    apply();
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
resize();
setMode('cruceta');
state.t = 0; apply();
view('corola');
const q = new URLSearchParams(location.search);
if (q.get('mode')) setMode(q.get('mode'), q.get('pose'));
if (q.get('t')) { state.t = Math.min(+q.get('t'), T_TOTAL); $('slider').value = state.t; apply(); }
if (q.get('view')) view(q.get('view'));
window.__ready = true;
requestAnimationFrame(loop);
