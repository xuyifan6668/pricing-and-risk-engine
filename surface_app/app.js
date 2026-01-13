const canvas = document.getElementById("surfaceCanvas");
const ctx = canvas.getContext("2d");
let viewSize = { width: canvas.clientWidth, height: canvas.clientHeight };

const controls = {
  spot: document.getElementById("spot"),
  rate: document.getElementById("rate"),
  dividend: document.getElementById("dividend"),
  borrow: document.getElementById("borrow"),
  noise: document.getElementById("noise"),
  smooth: document.getElementById("smooth"),
  strikeCount: document.getElementById("strikeCount"),
  expiryCount: document.getElementById("expiryCount"),
  showSurface: document.getElementById("showSurface"),
  showWire: document.getElementById("showWire"),
  showPoints: document.getElementById("showPoints"),
  rotX: document.getElementById("rotX"),
  rotY: document.getElementById("rotY"),
  zoom: document.getElementById("zoom"),
  calibrate: document.getElementById("calibrate"),
  regen: document.getElementById("regen"),
  quoteInput: document.getElementById("quoteInput"),
  rmsError: document.getElementById("rmsError"),
  quoteCount: document.getElementById("quoteCount"),
  nodeCount: document.getElementById("nodeCount"),
};

const baseExpiries = [0.25, 0.5, 1, 2, 3];
let marketQuotes = [];
let surface = null;

function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function forwardPrice(spot, rate, dividend, borrow, t) {
  return spot * Math.exp((rate - dividend - borrow) * t);
}

function generateQuotes() {
  const spot = parseFloat(controls.spot.value);
  const rate = parseFloat(controls.rate.value);
  const dividend = parseFloat(controls.dividend.value);
  const borrow = parseFloat(controls.borrow.value);
  const noiseBp = parseFloat(controls.noise.value);
  const strikes = 7;

  marketQuotes = [];
  baseExpiries.forEach((t) => {
    const fwd = forwardPrice(spot, rate, dividend, borrow, t);
    const atm = 0.22 - 0.02 * Math.log(1 + t);
    const skew = -0.12 + -0.03 * Math.log(1 + t);
    const curvature = 0.18 - 0.03 * Math.log(1 + t);
    for (let i = 0; i < strikes; i += 1) {
      const k = fwd * (0.7 + 0.1 * i);
      const x = Math.log(k / fwd);
      const vol = atm + skew * x + 0.5 * curvature * x * x;
      const noisy = vol + randn() * (noiseBp / 10000);
      marketQuotes.push({ expiry: t, strike: k, iv: Math.max(noisy, 0.01) });
    }
  });
  controls.quoteInput.value = JSON.stringify(marketQuotes, null, 2);
}

function parseQuotes() {
  try {
    const raw = JSON.parse(controls.quoteInput.value);
    marketQuotes = raw.filter((q) => q.expiry && q.strike && q.iv);
  } catch (err) {
    return false;
  }
  return true;
}

function fitQuadratic(xs, ys) {
  const n = xs.length;
  let s00 = n;
  let s10 = 0;
  let s20 = 0;
  let s30 = 0;
  let s40 = 0;
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  for (let i = 0; i < n; i += 1) {
    const x = xs[i];
    const x2 = x * x;
    s10 += x;
    s20 += x2;
    s30 += x2 * x;
    s40 += x2 * x2;
    b0 += ys[i];
    b1 += ys[i] * x;
    b2 += ys[i] * x2;
  }
  const det = s00 * (s20 * s40 - s30 * s30) - s10 * (s10 * s40 - s20 * s30) + s20 * (s10 * s30 - s20 * s20);
  if (Math.abs(det) < 1e-12) return [0.2, 0, 0];
  const inv = [
    (s20 * s40 - s30 * s30) / det,
    (s20 * s30 - s10 * s40) / det,
    (s10 * s30 - s20 * s20) / det,
    (s20 * s30 - s10 * s40) / det,
    (s00 * s40 - s20 * s20) / det,
    (s10 * s20 - s00 * s30) / det,
    (s10 * s30 - s20 * s20) / det,
    (s10 * s20 - s00 * s30) / det,
    (s00 * s20 - s10 * s10) / det,
  ];
  const a = inv[0] * b0 + inv[1] * b1 + inv[2] * b2;
  const b = inv[3] * b0 + inv[4] * b1 + inv[5] * b2;
  const c = inv[6] * b0 + inv[7] * b1 + inv[8] * b2;
  return [a, b, c];
}

function calibrateSurface() {
  const spot = parseFloat(controls.spot.value);
  const rate = parseFloat(controls.rate.value);
  const dividend = parseFloat(controls.dividend.value);
  const borrow = parseFloat(controls.borrow.value);
  const smoothing = parseFloat(controls.smooth.value);

  const quotesByExpiry = {};
  marketQuotes.forEach((q) => {
    if (!quotesByExpiry[q.expiry]) quotesByExpiry[q.expiry] = [];
    quotesByExpiry[q.expiry].push(q);
  });

  const expiries = Object.keys(quotesByExpiry)
    .map((t) => parseFloat(t))
    .sort((a, b) => a - b);

  const coeffs = expiries.map((t) => {
    const quotes = quotesByExpiry[t];
    const fwd = forwardPrice(spot, rate, dividend, borrow, t);
    const xs = quotes.map((q) => Math.log(q.strike / fwd));
    const ys = quotes.map((q) => q.iv);
    return fitQuadratic(xs, ys);
  });

  for (let i = 1; i < coeffs.length; i += 1) {
    const prev = coeffs[i - 1];
    const cur = coeffs[i];
    coeffs[i] = cur.map((val, idx) => prev[idx] * smoothing + val * (1 - smoothing));
  }

  surface = {
    expiries,
    coeffs,
    spot,
    rate,
    dividend,
    borrow,
  };
}

function evalSurface(t, k) {
  if (!surface) return 0.2;
  const { expiries, coeffs, spot, rate, dividend, borrow } = surface;
  if (expiries.length === 0) return 0.2;
  let idx = 0;
  while (idx < expiries.length - 1 && expiries[idx + 1] < t) idx += 1;
  let t0 = expiries[idx];
  let t1 = expiries[Math.min(idx + 1, expiries.length - 1)];
  const w = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
  const c0 = coeffs[idx];
  const c1 = coeffs[Math.min(idx + 1, coeffs.length - 1)];
  const c = c0.map((v, j) => v + (c1[j] - v) * w);
  const fwd = forwardPrice(spot, rate, dividend, borrow, t);
  const x = Math.log(k / fwd);
  return Math.max(c[0] + c[1] * x + c[2] * x * x, 0.01);
}

function computeRmsError() {
  if (!surface || marketQuotes.length === 0) return 0;
  let err = 0;
  marketQuotes.forEach((q) => {
    const fit = evalSurface(q.expiry, q.strike);
    err += (fit - q.iv) ** 2;
  });
  return Math.sqrt(err / marketQuotes.length);
}

function colorFromVol(vol, vmin, vmax) {
  const t = Math.max(0, Math.min(1, (vol - vmin) / (vmax - vmin || 1)));
  const r = Math.round(41 + 190 * t);
  const g = Math.round(74 + 120 * t);
  const b = Math.round(155 + 20 * (1 - t));
  return `rgba(${r}, ${g}, ${b}, 0.65)`;
}

function project(point, bounds) {
  const angleX = parseFloat(controls.rotX.value);
  const angleY = parseFloat(controls.rotY.value);
  const zoom = parseFloat(controls.zoom.value);

  const nx = (point.x - bounds.xMin) / (bounds.xMax - bounds.xMin || 1);
  const ny = (point.y - bounds.yMin) / (bounds.yMax - bounds.yMin || 1);
  const nz = (point.z - bounds.zMin) / (bounds.zMax - bounds.zMin || 1);

  let x = (nx - 0.5) * 2;
  let y = (ny - 0.5) * 2;
  let z = (nz - 0.5) * 2;

  const cosY = Math.cos(angleY);
  const sinY = Math.sin(angleY);
  const x1 = x * cosY + z * sinY;
  const z1 = -x * sinY + z * cosY;

  const cosX = Math.cos(angleX);
  const sinX = Math.sin(angleX);
  const y1 = y * cosX - z1 * sinX;
  const z2 = y * sinX + z1 * cosX;

  const perspective = 2.2 / (2.2 + z2);
  const scale = Math.min(viewSize.width, viewSize.height) * 0.35 * zoom;

  return {
    x: viewSize.width * 0.5 + x1 * scale * perspective,
    y: viewSize.height * 0.55 + y1 * scale * perspective,
    depth: z2,
  };
}

function drawAxes(bounds) {
  ctx.font = "12px Trebuchet MS, Lucida Grande, Verdana, sans-serif";
  const axes = [
    { x: bounds.xMin, y: bounds.yMin, z: bounds.zMin, label: "Expiry" },
    { x: bounds.xMax, y: bounds.yMin, z: bounds.zMin, label: "Strike" },
    { x: bounds.xMin, y: bounds.yMin, z: bounds.zMax, label: "Vol" },
  ];
  const origin = project({ x: bounds.xMin, y: bounds.yMin, z: bounds.zMin }, bounds);
  ctx.strokeStyle = "rgba(30, 30, 30, 0.4)";
  ctx.fillStyle = "rgba(30, 30, 30, 0.6)";
  ctx.lineWidth = 1.2;
  axes.forEach((axis) => {
    const p = project(axis, bounds);
    ctx.beginPath();
    ctx.moveTo(origin.x, origin.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.fillText(axis.label, p.x + 4, p.y + 4);
  });
}

function renderSurface() {
  ctx.clearRect(0, 0, viewSize.width, viewSize.height);
  ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
  ctx.fillRect(0, 0, viewSize.width, viewSize.height);

  if (!surface) return;

  const strikeCount = parseInt(controls.strikeCount.value, 10);
  const expiryCount = parseInt(controls.expiryCount.value, 10);

  const strikes = [];
  const expiries = [];
  const spot = parseFloat(controls.spot.value);

  const kMin = spot * 0.7;
  const kMax = spot * 1.3;
  for (let i = 0; i < strikeCount; i += 1) {
    strikes.push(kMin + (kMax - kMin) * (i / (strikeCount - 1)));
  }
  const tMin = 0.1;
  const tMax = 3.0;
  for (let i = 0; i < expiryCount; i += 1) {
    expiries.push(tMin + (tMax - tMin) * (i / (expiryCount - 1)));
  }

  const grid = [];
  let vMin = 1e9;
  let vMax = -1e9;
  expiries.forEach((t) => {
    const row = [];
    strikes.forEach((k) => {
      const vol = evalSurface(t, k);
      vMin = Math.min(vMin, vol);
      vMax = Math.max(vMax, vol);
      row.push(vol);
    });
    grid.push(row);
  });

  const bounds = {
    xMin: expiries[0],
    xMax: expiries[expiries.length - 1],
    yMin: strikes[0],
    yMax: strikes[strikes.length - 1],
    zMin: vMin,
    zMax: vMax,
  };

  const triangles = [];
  for (let i = 0; i < expiries.length - 1; i += 1) {
    for (let j = 0; j < strikes.length - 1; j += 1) {
      const p00 = { x: expiries[i], y: strikes[j], z: grid[i][j] };
      const p10 = { x: expiries[i + 1], y: strikes[j], z: grid[i + 1][j] };
      const p01 = { x: expiries[i], y: strikes[j + 1], z: grid[i][j + 1] };
      const p11 = { x: expiries[i + 1], y: strikes[j + 1], z: grid[i + 1][j + 1] };
      triangles.push([p00, p10, p11]);
      triangles.push([p00, p11, p01]);
    }
  }

  const projected = triangles.map((tri) => {
    const pts = tri.map((p) => project(p, bounds));
    const depth = pts.reduce((acc, p) => acc + p.depth, 0) / 3;
    const avgVol = tri.reduce((acc, p) => acc + p.z, 0) / 3;
    return { pts, depth, avgVol };
  });

  projected.sort((a, b) => a.depth - b.depth);

  if (controls.showSurface.checked) {
    projected.forEach((tri) => {
      ctx.beginPath();
      ctx.moveTo(tri.pts[0].x, tri.pts[0].y);
      ctx.lineTo(tri.pts[1].x, tri.pts[1].y);
      ctx.lineTo(tri.pts[2].x, tri.pts[2].y);
      ctx.closePath();
      ctx.fillStyle = colorFromVol(tri.avgVol, vMin, vMax);
      ctx.fill();
    });
  }

  if (controls.showWire.checked) {
    ctx.strokeStyle = "rgba(30, 30, 30, 0.18)";
    ctx.lineWidth = 1;
    for (let i = 0; i < expiries.length; i += 1) {
      ctx.beginPath();
      for (let j = 0; j < strikes.length; j += 1) {
        const p = project({ x: expiries[i], y: strikes[j], z: grid[i][j] }, bounds);
        if (j === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    }
    for (let j = 0; j < strikes.length; j += 1) {
      ctx.beginPath();
      for (let i = 0; i < expiries.length; i += 1) {
        const p = project({ x: expiries[i], y: strikes[j], z: grid[i][j] }, bounds);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    }
  }

  if (controls.showPoints.checked) {
    ctx.fillStyle = "rgba(214, 107, 50, 0.85)";
    marketQuotes.forEach((q) => {
      const p = project({ x: q.expiry, y: q.strike, z: q.iv }, bounds);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  drawAxes(bounds);

  controls.rmsError.textContent = computeRmsError().toFixed(4);
  controls.quoteCount.textContent = `${marketQuotes.length}`;
  controls.nodeCount.textContent = `${expiries.length * strikes.length}`;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  viewSize = { width: rect.width, height: rect.height };
  renderSurface();
}

function boot() {
  resizeCanvas();
  generateQuotes();
  calibrateSurface();
  renderSurface();
}

controls.calibrate.addEventListener("click", () => {
  if (!parseQuotes()) return;
  calibrateSurface();
  renderSurface();
});

controls.regen.addEventListener("click", () => {
  generateQuotes();
  calibrateSurface();
  renderSurface();
});

["rotX", "rotY", "zoom", "strikeCount", "expiryCount", "showSurface", "showWire", "showPoints"].forEach((id) => {
  controls[id].addEventListener("input", renderSurface);
});

window.addEventListener("resize", resizeCanvas);

boot();
