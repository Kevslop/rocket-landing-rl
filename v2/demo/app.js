// app.js — v2 interactive demo: Three.js scene + SpaceX-style HUD.
// Sim is z-up (x, y, z); three.js is y-up: map (x, y, z) -> (x, z, y).
import * as THREE from 'three';

const WS_URL = `ws://${location.hostname}:8124`;
const $ = (id) => document.getElementById(id);

// ---------------- state ----------------
let latest = null;          // last sim state
let replay = null;          // {frames, i, t}
let mode = 'ai';
let camMode = 'follow';     // follow | pad | tower
let speed = 0.5;
let manualKeys = {};
let manual = { throttle: 0, tilt_x: 0, tilt_z: 0 };
let loggedEp = -1, loggedFuelEmpty = false, loggedPhase = 0;
let ws = null;

// ---------------- three setup ----------------
const container = $('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e1a);
scene.fog = new THREE.FogExp2(0x0a0e1a, 0.0016);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.5, 2000);
camera.position.set(40, 25, 40);

// dusk lighting
scene.add(new THREE.HemisphereLight(0x8a7bd8, 0x1a1410, 0.55));
const sun = new THREE.DirectionalLight(0xff9a5c, 1.1);
sun.position.set(-120, 40, 60);
scene.add(sun);
const padLight = new THREE.SpotLight(0xbfd9ff, 800, 120, 0.5, 0.4);
padLight.position.set(0, 40, 0);
scene.add(padLight);

// stars
{
  const n = 900, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 700 + Math.random() * 500, th = Math.random() * Math.PI * 2;
    pos[i * 3] = r * Math.cos(th); pos[i * 3 + 1] = 150 + Math.random() * 500;
    pos[i * 3 + 2] = r * Math.sin(th);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xaFC4e8, size: 1.6, sizeAttenuation: false, transparent: true, opacity: 0.7 })));
}

// ground
{
  const gnd = new THREE.Mesh(
    new THREE.CircleGeometry(400, 64),
    new THREE.MeshStandardMaterial({ color: 0x11141c, roughness: 1 })
  );
  gnd.rotation.x = -Math.PI / 2;
  scene.add(gnd);
  const grid = new THREE.GridHelper(400, 40, 0x2a3a55, 0x1a2438);
  grid.position.y = 0.02;
  scene.add(grid);
}

// ---------------- launch pad + tower (origin) ----------------
function discPad(r, color) {
  const grp = new THREE.Group();
  const disc = new THREE.Mesh(
    new THREE.CylinderGeometry(r, r, 0.6, 48),
    new THREE.MeshStandardMaterial({ color: 0x232830, roughness: 0.9 })
  );
  disc.position.y = 0.3;
  grp.add(disc);
  // target rings via canvas texture
  const cv = document.createElement('canvas'); cv.width = cv.height = 256;
  const cx = cv.getContext('2d');
  cx.fillStyle = '#232830'; cx.fillRect(0, 0, 256, 256);
  cx.strokeStyle = color; cx.lineWidth = 6;
  for (const rr of [110, 78, 46]) { cx.beginPath(); cx.arc(128, 128, rr, 0, 7); cx.stroke(); }
  cx.fillStyle = color; cx.beginPath(); cx.arc(128, 128, 12, 0, 7); cx.fill();
  const tex = new THREE.CanvasTexture(cv);
  const top = new THREE.Mesh(
    new THREE.CircleGeometry(r * 0.98, 48),
    new THREE.MeshBasicMaterial({ map: tex })
  );
  top.rotation.x = -Math.PI / 2; top.position.y = 0.62;
  grp.add(top);
  // beacon light column
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.35, 0.9, 26, 12, 1, true),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false })
  );
  beam.position.y = 13.6;
  grp.add(beam);
  return grp;
}
const launchPad = discPad(7, '#4da6ff');
scene.add(launchPad);

// Mechazilla-ish tower
{
  const tower = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0x3a4150, roughness: 0.7, metalness: 0.5 });
  const h = 70;
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(1.2, h, 1.2), mat);
    leg.position.set(sx * 3, h / 2, sz * 3);
    tower.add(leg);
  }
  for (let y = 6; y < h; y += 8) {
    const ring = new THREE.Mesh(new THREE.BoxGeometry(7.4, 1, 7.4), mat);
    ring.position.y = y;
    tower.add(ring);
  }
  // chopstick arms
  for (const s of [-1, 1]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(2, 1.6, 16), mat);
    arm.position.set(s * 5.5, 52, 6);
    tower.add(arm);
  }
  const lamp = new THREE.PointLight(0xffd9a0, 600, 90);
  lamp.position.set(0, h + 2, 0);
  tower.add(lamp);
  tower.position.set(22, 0, -14);
  scene.add(tower);
}

// landing pad (moves + tilts per episode)
const landPad = discPad(7, '#3ddc84');
scene.add(landPad);

// waypoint marker: pulsing diamond
const wpMarker = new THREE.Mesh(
  new THREE.OctahedronGeometry(2.2),
  new THREE.MeshBasicMaterial({ color: 0x4da6ff, wireframe: true, transparent: true, opacity: 0.9 })
);
scene.add(wpMarker);

// ---------------- rocket ----------------
const rocket = new THREE.Group();
{
  const white = new THREE.MeshStandardMaterial({ color: 0xdfe6ee, roughness: 0.45, metalness: 0.25 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x22262e, roughness: 0.6, metalness: 0.4 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.1, 9, 24), white);
  body.position.y = 4.5; rocket.add(body);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(1.1, 2.6, 24), dark);
  nose.position.y = 10.3; rocket.add(nose);
  for (let i = 0; i < 4; i++) {
    const fin = new THREE.Mesh(new THREE.BoxGeometry(0.12, 2.4, 1.4), dark);
    const a = i * Math.PI / 2;
    fin.position.set(Math.cos(a) * 1.35, 1.4, Math.sin(a) * 1.35);
    fin.rotation.y = -a;
    rocket.add(fin);
  }
  const bell = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 1.0, 1.2, 16, 1, true), dark);
  bell.position.y = -0.4; rocket.add(bell);
  // engine glow sprite
  const cv = document.createElement('canvas'); cv.width = cv.height = 128;
  const cx = cv.getContext('2d');
  const grad = cx.createRadialGradient(64, 64, 4, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,240,200,1)');
  grad.addColorStop(0.35, 'rgba(255,170,60,0.85)');
  grad.addColorStop(1, 'rgba(255,120,20,0)');
  cx.fillStyle = grad; cx.fillRect(0, 0, 128, 128);
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(cv), transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending
  }));
  glow.position.y = -1.6; glow.name = 'glow';
  rocket.add(glow);
  const el = new THREE.PointLight(0xff9a3c, 0, 60);
  el.position.y = -2; el.name = 'elight';
  rocket.add(el);
}
scene.add(rocket);

// exhaust plume particles
const PLUME_N = 420;
const plumeGeo = new THREE.BufferGeometry();
const plumePos = new Float32Array(PLUME_N * 3);
const plumeVel = new Float32Array(PLUME_N * 3);
const plumeLife = new Float32Array(PLUME_N);
plumeGeo.setAttribute('position', new THREE.BufferAttribute(plumePos, 3));
const plume = new THREE.Points(plumeGeo, new THREE.PointsMaterial({
  color: 0xffb050, size: 1.5, transparent: true, opacity: 0.85,
  blending: THREE.AdditiveBlending, depthWrite: false
}));
plume.frustumCulled = false;
scene.add(plume);
let plumeIdx = 0;
function spawnPlume(origin, throttle, dt) {
  const n = Math.floor(throttle * 240 * dt) + (Math.random() < throttle * 4 * dt ? 1 : 0);
  for (let k = 0; k < n; k++) {
    const i = plumeIdx = (plumeIdx + 1) % PLUME_N;
    plumePos[i * 3] = origin.x + (Math.random() - 0.5) * 0.8;
    plumePos[i * 3 + 1] = origin.y;
    plumePos[i * 3 + 2] = origin.z + (Math.random() - 0.5) * 0.8;
    plumeVel[i * 3] = (Math.random() - 0.5) * 6;
    plumeVel[i * 3 + 1] = -(14 + Math.random() * 14) * (0.4 + throttle);
    plumeVel[i * 3 + 2] = (Math.random() - 0.5) * 6;
    plumeLife[i] = 0.5 + Math.random() * 0.5;
  }
}
function tickPlume(dt) {
  for (let i = 0; i < PLUME_N; i++) {
    if (plumeLife[i] <= 0) { plumePos[i * 3 + 1] = -100; continue; }
    plumeLife[i] -= dt;
    plumePos[i * 3] += plumeVel[i * 3] * dt;
    plumePos[i * 3 + 1] += plumeVel[i * 3 + 1] * dt;
    plumePos[i * 3 + 2] += plumeVel[i * 3 + 2] * dt;
  }
  plumeGeo.attributes.position.needsUpdate = true;
}

// trajectory trail
const TRAIL_N = 500;
const trailGeo = new THREE.BufferGeometry();
const trailPos = new Float32Array(TRAIL_N * 3);
const trailCol = new Float32Array(TRAIL_N * 3);
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 }));
trail.frustumCulled = false;
scene.add(trail);
let trailPts = [];
function pushTrail(p) {
  trailPts.push(p.clone());
  if (trailPts.length > TRAIL_N) trailPts.shift();
  for (let i = 0; i < TRAIL_N; i++) {
    const q = trailPts[Math.min(i, trailPts.length - 1)] || p;
    trailPos[i * 3] = q.x; trailPos[i * 3 + 1] = q.y; trailPos[i * 3 + 2] = q.z;
    const f = i / TRAIL_N;
    trailCol[i * 3] = 0.3 * f + 0.05; trailCol[i * 3 + 1] = 0.65 * f + 0.05; trailCol[i * 3 + 2] = f + 0.05;
  }
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate = true;
}

// precipitation (rain / snow)
function makePrecip(count, color, size) {
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 260;
    pos[i * 3 + 1] = Math.random() * 120;
    pos[i * 3 + 2] = (Math.random() - 0.5) * 260;
  }
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const p = new THREE.Points(g, new THREE.PointsMaterial({ color, size, transparent: true, opacity: 0.55, depthWrite: false }));
  p.frustumCulled = false; p.visible = false;
  p.userData = { pos, count };
  scene.add(p);
  return p;
}
const rain = makePrecip(900, 0x8fb8dd, 0.7);
const snow = makePrecip(700, 0xffffff, 1.1);
function tickPrecip(p, dt, fall, wind) {
  const { pos, count } = p.userData;
  for (let i = 0; i < count; i++) {
    pos[i * 3 + 1] -= fall * dt * (0.7 + Math.random() * 0.6);
    pos[i * 3] += wind[0] * dt * 0.5;
    pos[i * 3 + 2] += wind[1] * dt * 0.5;
    if (pos[i * 3 + 1] < 0) {
      pos[i * 3 + 1] = 120;
      pos[i * 3] = (Math.random() - 0.5) * 260;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 260;
    }
  }
  p.geometry.attributes.position.needsUpdate = true;
}

// wind arrow at landing pad
const windArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 12, 0x4da6ff, 3, 2);
scene.add(windArrow);

// ---------------- HUD ----------------
function fmtT(t) {
  const m = Math.floor(t / 60), s = t % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}
function setStatus(txt, cls) {
  const el = $('status');
  el.textContent = txt;
  el.className = cls;
}
function log(msg, cls) {
  const el = $('log');
  const d = document.createElement('div');
  if (cls) d.className = cls;
  d.textContent = msg;
  el.prepend(d);
  while (el.children.length > 6) el.lastChild.remove();
}

function updateHUD(s) {
  const [x, y, z] = s.pos, [vx, vy, vz] = s.vel;
  const vxy = Math.hypot(vx, vy);
  $('tval').textContent = fmtT(s.t);
  $('alt').innerHTML = `${z.toFixed(1)}<small>m</small>`;
  const ve = $('vspd');
  ve.innerHTML = `${vz >= 0 ? '+' : ''}${vz.toFixed(1)}<small>m/s</small>`;
  ve.className = 'val ' + (vz < -12 ? 'neg' : 'pos');
  $('hspd').innerHTML = `${vxy.toFixed(1)}<small>m/s</small>`;
  $('drng').innerHTML = `${Math.hypot(x, y).toFixed(1)}<small>m</small>`;
  $('tilt').innerHTML = `${(Math.hypot(s.tilt[0], s.tilt[1]) * 57.3).toFixed(1)}<small>&deg;</small>`;

  // pad distance / bearing
  const dx = s.pad.x - x, dy = s.pad.y - y;
  const pd = Math.hypot(dx, dy);
  const brg = ((Math.atan2(dx, dy) * 57.2958) + 360) % 360;
  $('paddist').innerHTML = `${pd.toFixed(0)}<small>m</small>`;
  $('padbrg').innerHTML = `${brg.toFixed(0).padStart(3, '0')}<small>&deg;</small>`;
  $('padslope').innerHTML = `${(Math.hypot(s.pad.slope[0], s.pad.slope[1]) * 100).toFixed(1)}<small>%</small>`;

  $('fuelv').innerHTML = `${(s.fuel * 100).toFixed(0)}<small>%</small>`;
  const ff = $('fuel-fill');
  ff.style.width = `${s.fuel * 100}%`;
  ff.className = s.fuel < 0.2 ? 'low' : '';
  $('thrv').innerHTML = `${(s.throttle * 100).toFixed(0)}<small>%</small>`;
  $('thr-fill').style.width = `${s.throttle * 100}%`;
  const lit = Math.ceil(s.throttle * 3 - 1e-6);
  for (let i = 1; i <= 3; i++) $('eng' + i).className = 'eng' + (i <= lit ? ' on' : '');

  const wv = Math.hypot(s.wind[0], s.wind[1]);
  $('windv').textContent = wv.toFixed(1);
  $('wind-arrow').style.transform = `rotate(${Math.atan2(s.wind[1], s.wind[0]) * 57.3 - 90}deg)`;

  // mission rows
  const ph = s.phase === 0 ? 'ASCENT' : 'LANDING';
  $('phasev').textContent = s.mode === 'manual' ? ph + ' · MANUAL' : ph;
  const eng = s.engine;
  $('enginev').textContent = `${eng.name} · ${eng.tech}`;
  if (eng.projected) $('enginev').innerHTML += ' <span class="proj">PROJECTED</span>';
  $('pcv').innerHTML = `${eng.pc_bar.toFixed(0)}<small>bar</small>`;
  $('pc-fill').style.width = `${Math.min(100, eng.pc_bar / 4)}%`;
  const detRow = $('det-row');
  if (eng.det_khz) {
    detRow.style.display = 'flex';
    $('detv').innerHTML = `${eng.det_khz.toFixed(2)}<small>kHz</small>`;
  } else detRow.style.display = 'none';
  $('weatherv').textContent = s.weather.toUpperCase();
  $('massv').innerHTML = `${eng.tw.toFixed(2)}<small>T/W</small>`;

  // status
  if (s.done) {
    if (s.outcome === 'landed') setStatus('TOUCHDOWN ✓', 'nominal');
    else setStatus(s.outcome === 'timeout' ? 'TIMEOUT' : 'VEHICLE LOST', 'bad');
  } else if (s.fuel <= 0.005 && s.t > 2) setStatus('FUEL DEPLETED', 'warn');
  else if (s.phase === 0) setStatus(s.mode === 'manual' ? 'MANUAL · ASCENT' : 'ASCENT', 'nominal');
  else setStatus(s.mode === 'manual' ? 'MANUAL · LANDING' : 'TERMINAL DESCENT', 'nominal');
}

// ---------------- scene update ----------------
const V3 = (x, y, z) => new THREE.Vector3(x, z, y); // sim -> three
function updateScene(s, dt) {
  const sc = s.engine.scale;
  rocket.scale.setScalar(sc);
  rocket.position.copy(V3(...s.pos));
  rocket.position.y = Math.max(rocket.position.y, 0.6); // rest on pad surface
  rocket.rotation.set(s.tilt[1], 0, -s.tilt[0]);

  // glow + plume
  const glow = rocket.getObjectByName('glow');
  const el = rocket.getObjectByName('elight');
  glow.material.opacity = s.throttle * 0.95;
  const gs = (2 + s.throttle * 7) * sc;
  glow.scale.set(gs, gs, 1);
  el.intensity = s.throttle * 900 * sc;
  const nozzle = new THREE.Vector3();
  rocket.getObjectByName('glow').getWorldPosition(nozzle);
  if (s.throttle > 0.02) spawnPlume(nozzle, s.throttle, dt);
  tickPlume(dt);
  pushTrail(rocket.position);

  // landing pad pose
  landPad.position.set(s.pad.x, 0, s.pad.y);
  const n = new THREE.Vector3(-s.pad.slope[0], 1, -s.pad.slope[1]).normalize();
  landPad.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), n);

  // waypoint marker
  wpMarker.position.copy(V3(...s.waypoint));
  wpMarker.visible = s.phase === 0 && !s.done;
  const puls = 1 + 0.18 * Math.sin(performance.now() / 280);
  wpMarker.scale.setScalar(puls);
  wpMarker.rotation.y += dt * 1.2;

  // wind arrow at pad
  windArrow.position.set(s.pad.x, 2, s.pad.y);
  const wv = Math.hypot(s.wind[0], s.wind[1]);
  if (wv > 0.3) {
    windArrow.setDirection(new THREE.Vector3(s.wind[0] / wv, 0, s.wind[1] / wv));
    windArrow.setLength(Math.min(4 + wv, 16));
    windArrow.visible = true;
  } else windArrow.visible = false;

  // weather visuals
  rain.visible = s.weather === 'rain';
  snow.visible = s.weather === 'snow';
  if (rain.visible) tickPrecip(rain, dt, 55, s.wind);
  if (snow.visible) tickPrecip(snow, dt, 4, s.wind);
  const fogD = { clear: 0.0016, rain: 0.004, snow: 0.007, ice: 0.0022 }[s.weather] ?? 0.0016;
  scene.fog.density += (fogD - scene.fog.density) * Math.min(1, dt * 2);

  // camera
  const rp = rocket.position;
  let desired, look;
  if (camMode === 'pad') {
    desired = new THREE.Vector3(s.pad.x + 26, 20, s.pad.y + 26);
    look = new THREE.Vector3(s.pad.x, 4, s.pad.y);
  } else if (camMode === 'tower') {
    desired = new THREE.Vector3(-30, 26, 30);
    look = new THREE.Vector3(0, 18, 0);
  } else {
    desired = new THREE.Vector3(rp.x + 30, rp.y + 16, rp.z + 30);
    look = rp.clone();
  }
  camera.position.lerp(desired, 1 - Math.pow(0.02, dt));
  camera.lookAt(look);
}

// ---------------- replay ----------------
function startReplay(frames) {
  if (!frames || !frames.length) return;
  replay = { frames, i: 0 };
  $('replay-banner').style.display = 'block';
  $('btn-replay').classList.add('active');
  log('REPLAY ×0.3 — last flight', '');
}
function stopReplay() {
  replay = null;
  $('replay-banner').style.display = 'none';
  $('btn-replay').classList.remove('active');
}
function tickReplay(dt) {
  // 10 Hz frames, play at 0.3x -> advance 3 frames/sec
  replay.i += dt * 3;
  if (replay.i >= replay.frames.length - 1) { stopReplay(); return; }
  const f = replay.frames[Math.floor(replay.i)];
  const s = {
    ...latest, t: f.t, pos: f.pos, tilt: f.tilt, throttle: f.throttle,
    fuel: f.fuel, phase: f.phase, vel: latest.vel, wind: latest.wind,
  };
  updateHUD(s);
  updateScene(s, dt);
}

// ---------------- websocket ----------------
let lastFrames = null, lastReport = null;
function connect() {
  $('conn').style.display = 'block';
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { $('conn').style.display = 'none'; };
  ws.onclose = () => { setTimeout(connect, 1500); $('conn').style.display = 'block'; };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'hello') {
      mode = m.mode; speed = m.speed ?? 0.5;
      syncSpeedBtns(); syncModeBtns();
      return;
    }
    if (m.type === 'loadouts') { fillLoadouts(m); return; }
    if (m.type !== 'state') return;
    const isNewEp = latest && m.t < latest.t - 0.5;
    latest = m;
    if (isNewEp) {
      stopReplay();
      trailPts = [];
      loggedFuelEmpty = false; loggedPhase = 0;
      $('report').style.display = 'none';
      log(`T+0.0 — NEW FLIGHT · ${m.engine.name} + ${m.engine.tech}${m.engine.projected ? ' [PROJECTED]' : ''} · ${m.weather.toUpperCase()}`, '');
    }
    if (m.phase === 1 && loggedPhase === 0) {
      loggedPhase = 1;
      log(`T+${m.t.toFixed(1)} — WAYPOINT REACHED, heading to pad`, 'good');
    }
    if (m.fuel <= 0.005 && !loggedFuelEmpty && m.t > 2) {
      loggedFuelEmpty = true;
      log(`T+${m.t.toFixed(1)} — PROPELLANT DEPLETED`, 'bad');
    }
    if (m.done && m.report && m.report !== lastReport) {
      lastReport = m.report;
      lastFrames = m.trajectory;
      const r = m.report;
      if (r.outcome === 'landed') log(`T+${r.flight_time}s — TOUCHDOWN ✓ · fuel used ${r.fuel_used_pct}%`, 'good');
      else log(`T+${r.flight_time}s — ${r.outcome.toUpperCase()}${r.reason ? ' (' + r.reason + ')' : ''}`, 'bad');
      showReport(r, m);
      $('btn-replay').disabled = false;
    }
  };
}

// ---------------- controls ----------------
function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
$('btn-ai').onclick = () => { mode = 'ai'; send({ cmd: 'new_episode', mode: 'ai' }); syncModeBtns(); };
$('btn-manual').onclick = () => { mode = 'manual'; send({ cmd: 'new_episode', mode: 'manual' }); syncModeBtns(); };
$('btn-rand').onclick = () => send({ cmd: 'new_episode', seed: Math.floor(Math.random() * 1e9) });
$('btn-auto').onclick = (e) => {
  const b = e.currentTarget, on = !b.classList.contains('active');
  b.classList.toggle('active', on);
  send({ cmd: 'auto', on });
};
$('btn-replay').onclick = () => {
  if (replay) stopReplay();
  else if (lastFrames) startReplay(lastFrames);
};
$('btn-cam').onclick = () => {
  camMode = camMode === 'follow' ? 'pad' : camMode === 'pad' ? 'tower' : 'follow';
  $('btn-cam').innerHTML = `🎥 ${camMode.toUpperCase()}`;
};
function syncModeBtns() {
  $('btn-ai').classList.toggle('active', mode === 'ai');
  $('btn-manual').classList.toggle('active', mode === 'manual');
}
// speed segmented control
for (const b of document.querySelectorAll('#speed button')) {
  b.onclick = () => { speed = parseFloat(b.dataset.v); send({ cmd: 'speed', value: speed }); syncSpeedBtns(); };
}
function syncSpeedBtns() {
  for (const b of document.querySelectorAll('#speed button'))
    b.classList.toggle('active', Math.abs(parseFloat(b.dataset.v) - speed) < 1e-9);
}
// loadout picker
function fillLoadouts(m) {
  const es = $('sel-engine'), ts = $('sel-tech'), ws_ = $('sel-weather');
  es.innerHTML = m.engines.map(e => `<option value="${e.id}">${e.name} — ${e.maker}</option>`).join('');
  ts.innerHTML = m.techs.map(t => `<option value="${t.id}">${t.name}${t.projected ? ' [PROJECTED]' : ''}</option>`).join('');
  ws_.innerHTML = `<option value="auto">Auto (random)</option>` +
    m.weather.map(w => `<option value="${w.id}">${w.name}</option>`).join('');
  const upd = () => {
    const e = m.engines.find(x => x.id === es.value);
    const t = m.techs.find(x => x.id === ts.value);
    $('loadout-spec').innerHTML =
      `<b>${e.name}</b> (${e.maker})<br>${e.blurb}<br><b>${t.name}</b>${t.projected ? ' <span class="proj">PROJECTED</span>' : ''}<br>${t.blurb}`;
  };
  es.onchange = upd; ts.onchange = upd; upd();
}
$('btn-fly').onclick = () => {
  send({ cmd: 'set_loadout', engine: $('sel-engine').value, tech: $('sel-tech').value, weather: $('sel-weather').value });
  mode = 'ai'; syncModeBtns();
};
$('report-close').onclick = () => { $('report').style.display = 'none'; };
function showReport(r, m) {
  const ok = r.outcome === 'landed';
  $('report-title').textContent = ok ? 'MISSION SUCCESS ✓' : 'MISSION FAILED';
  $('report-title').className = ok ? 'good' : 'bad';
  $('report-body').innerHTML =
    `<div><span>LOADOUT</span><b>${m.engine.name} + ${m.engine.tech}${m.engine.projected ? ' [P]' : ''}</b></div>
     <div><span>WEATHER</span><b>${m.weather.toUpperCase()} · pad μ=${m.pad.mu}</b></div>
     <div><span>WAYPOINT</span><b>${r.waypoint_reached ? 'REACHED ✓' : 'MISSED'}</b></div>
     <div><span>OUTCOME</span><b>${r.outcome.toUpperCase()}${r.reason ? ' (' + r.reason + ')' : ''}</b></div>
     <div><span>FUEL USED</span><b>${r.fuel_used_pct}%</b></div>
     <div><span>FLIGHT TIME</span><b>${r.flight_time}s</b></div>`;
  $('report').style.display = 'block';
}

// manual keyboard
addEventListener('keydown', (e) => {
  if (e.repeat) return;
  manualKeys[e.code] = true;
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
});
addEventListener('keyup', (e) => { manualKeys[e.code] = false; });
setInterval(() => {
  if (mode !== 'manual' || !latest) return;
  // camera-relative tilt
  const rp = rocket.position, cp = camera.position;
  const fx = rp.x - cp.x, fz = rp.z - cp.z;
  const fl = Math.hypot(fx, fz) || 1;
  const fwd = [fx / fl, fz / fl], rgt = [-fwd[1], fwd[0]];
  let ix = 0, iy = 0;
  if (manualKeys.KeyW || manualKeys.ArrowUp) { ix += fwd[0]; iy += fwd[1]; }
  if (manualKeys.KeyS || manualKeys.ArrowDown) { ix -= fwd[0]; iy -= fwd[1]; }
  if (manualKeys.KeyD || manualKeys.ArrowRight) { ix += rgt[0]; iy += rgt[1]; }
  if (manualKeys.KeyA || manualKeys.ArrowLeft) { ix -= rgt[0]; iy -= rgt[1]; }
  const il = Math.hypot(ix, iy) || 1;
  const g = (latest.engine && latest.engine.gimbal_deg ? latest.engine.gimbal_deg : 8) * Math.PI / 180;
  manual.tilt_x = (ix / il) * Math.min(1, Math.hypot(ix, iy)) * g * (ix || iy ? 1 : 0);
  manual.tilt_z = (iy / il) * Math.min(1, Math.hypot(ix, iy)) * g * (ix || iy ? 1 : 0);
  if (manualKeys.KeyR) manual.throttle = Math.min(1, manual.throttle + 0.04);
  if (manualKeys.KeyF) manual.throttle = Math.max(0, manual.throttle - 0.04);
  if (manualKeys.KeyX) manual.throttle = 0;
  send({ cmd: 'manual', ...manual });
}, 50);

// ---------------- main loop ----------------
let lastT = performance.now();
function loop() {
  requestAnimationFrame(loop);
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  if (replay) tickReplay(dt);
  else if (latest) { updateHUD(latest); updateScene(latest, dt); }
  renderer.render(scene, camera);
}
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
syncModeBtns(); syncSpeedBtns();
connect();
loop();
