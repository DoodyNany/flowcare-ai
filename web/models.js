/* Flow Care AI - in-browser inference. No server, no outside AI service.
   FC.forecast   : gradient-boosted trees (bed census 8 h ahead, ED 1-7 days ahead)
   FC.risk       : logistic regression (deterioration risk, warn-only)
   FC.xray       : small CNN (pneumonia pattern flag + heat-map, warn-only)        */
var FC = (function () {
  "use strict";

  // ---------- decision-tree ensembles ----------
  function treeSum(model, x) {
    var s = model.init;
    for (var k = 0; k < model.trees.length; k++) {
      var t = model.trees[k], n = 0;
      while (t.l[n] !== -1) n = (x[t.f[n]] <= t.t[n]) ? t.l[n] : t.r[n];
      s += t.v[n];
    }
    return s;
  }

  // calendar helpers (data is naive Riyadh local time)
  var DAY = 86400000;
  function parseNaive(s) { // "2025-12-24 00:00:00" or "2025-12-24" -> ms as if UTC
    var p = s.split(/[- :T]/).map(Number);
    return Date.UTC(p[0], p[1] - 1, p[2], p[3] || 0, p[4] || 0);
  }
  function makeCal(cal) {
    var t0 = parseNaive(cal.start);
    return function (ms) {
      var i = Math.floor((ms - t0) / DAY);
      return { ramadan: cal.ramadan[i] || 0, hajj: cal.hajj[i] || 0, eid: cal.eid[i] || 0,
               h_day_dh: cal.h_day_dh[i] || 0, ok: i >= 0 && i < cal.ramadan.length };
    };
  }
  function pyDow(ms) { return (new Date(ms).getUTCDay() + 6) % 7; } // Mon=0 like pandas
  function isWinter(ms) { var m = new Date(ms).getUTCMonth() + 1; return (m === 11 || m === 12 || m === 1 || m === 2) ? 1 : 0; }

  function hourlyFeatures(c, a, i, ms, calf) {
    var cf = calf(ms), s3 = 0, s24 = 0;
    for (var j = 0; j < 24; j++) { if (j < 3) s3 += a[i - j]; s24 += a[i - j]; }
    return [c[i], c[i - 1], c[i - 3], c[i - 6], c[i - 24], c[i - 168], s3, s24,
            new Date(ms).getUTCHours(), pyDow(ms), cf.ramadan, cf.hajj, cf.eid, isWinter(ms)];
  }

  function Forecaster(M) {
    this.M = M; this.calf = makeCal(M.calendar);
    this.t0 = parseNaive(M.replay.start);
    this.dailyT0 = parseNaive(M.daily_ed.start);
  }
  Forecaster.prototype.census8h = function (dept, census, arrivals, i) {
    var ms = this.t0 + i * 3600000;
    var x = hourlyFeatures(census, arrivals, i, ms, this.calf);
    return x[0] + treeSum(this.M.depts[dept], x);
  };
  Forecaster.prototype.edWeek = function (dayMs, visits) {
    // forecast ED visits for the 7 days after dayMs, using history up to dayMs
    var i = Math.round((dayMs - this.dailyT0) / DAY), out = [];
    var m7 = 0, m28 = 0;
    for (var j = 0; j < 28; j++) { if (j < 7) m7 += visits[i - j]; m28 += visits[i - j]; }
    m7 /= 7; m28 /= 28;
    for (var k = 1; k <= 7; k++) {
      var t = dayMs + k * DAY, cf = this.calf(t);
      var x = [k, m7, m28, pyDow(t), isWinter(t), cf.ramadan, cf.hajj, cf.eid, cf.h_day_dh];
      out.push({ ms: t, pred: treeSum(this.M.ed_daily, x), actual: visits[i + k], cal: cf });
    }
    return out;
  };

  // ---------- deterioration risk ----------
  var RISK_FEATS = [
    function (r) { return Math.max(0, r.age - 50) / 10; },
    function (r) { return Math.max(0, r.hr - 90) / 10; },
    function (r) { return Math.max(0, 50 - r.hr) / 10; },
    function (r) { return Math.max(0, r.rr - 20) / 2; },
    function (r) { return Math.max(0, 110 - r.sbp) / 10; },
    function (r) { return Math.max(0, 96 - r.spo2); },
    function (r) { return r.on_o2 ? 1 : 0; },
    function (r) { return Math.max(0, r.temp - 38); },
    function (r) { return Math.max(0, 36 - r.temp); },
    function (r) { return r.confused ? 1 : 0; },
    function (r) { return Math.max(0, r.lactate - 2); },
    function (r) { return Math.max(0, r.wbc - 11) / 2 + Math.max(0, 4 - r.wbc) / 2; },
    function (r) { return Math.max(0, r.creat - 110) / 50; }
  ];
  function risk(RM, r) {
    var z = RM.intercept, contrib = [];
    for (var k = 0; k < RISK_FEATS.length; k++) {
      var c = RM.coef[k] * RISK_FEATS[k](r);
      z += c; contrib.push({ name: RM.features[k], value: c });
    }
    contrib.sort(function (a, b) { return b.value - a.value; });
    var p = 1 / (1 + Math.exp(-z));
    var level = p >= RM.alert_threshold ? "alert" : (p >= RM.watch_threshold ? "watch" : "low");
    return { p: p, level: level, contrib: contrib };
  }
  function news2(r) {
    var s = 0, v;
    v = r.rr;   s += v <= 8 ? 3 : v <= 11 ? 1 : v <= 20 ? 0 : v <= 24 ? 2 : 3;
    v = r.spo2; s += v <= 91 ? 3 : v <= 93 ? 2 : v <= 95 ? 1 : 0;
    s += r.on_o2 ? 2 : 0;
    v = r.sbp;  s += v <= 90 ? 3 : v <= 100 ? 2 : v <= 110 ? 1 : v <= 219 ? 0 : 3;
    v = r.hr;   s += v <= 40 ? 3 : v <= 50 ? 1 : v <= 90 ? 0 : v <= 110 ? 1 : v <= 130 ? 2 : 3;
    s += r.confused ? 3 : 0;
    v = r.temp; s += v <= 35 ? 3 : v <= 36 ? 1 : v <= 38 ? 0 : v <= 39 ? 1 : 2;
    return s;
  }

  // ---------- CNN ----------
  function conv3(inp, H, W, C, w, b, O) { // SAME padding, HWIO weights, ReLU
    var out = new Float32Array(H * W * O);
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
      var ob = (y * W + x) * O;
      for (var o = 0; o < O; o++) out[ob + o] = b[o];
      for (var ky = 0; ky < 3; ky++) {
        var yy = y + ky - 1; if (yy < 0 || yy >= H) continue;
        for (var kx = 0; kx < 3; kx++) {
          var xx = x + kx - 1; if (xx < 0 || xx >= W) continue;
          var ib = (yy * W + xx) * C, wb = (ky * 3 + kx) * C * O;
          for (var c = 0; c < C; c++) {
            var v = inp[ib + c]; if (v === 0) continue;
            var wo = wb + c * O;
            for (var o2 = 0; o2 < O; o2++) out[ob + o2] += v * w[wo + o2];
          }
        }
      }
      for (var o3 = 0; o3 < O; o3++) if (out[ob + o3] < 0) out[ob + o3] = 0;
    }
    return out;
  }
  function pool2(inp, H, W, C) {
    var h = H / 2, w = W / 2, out = new Float32Array(h * w * C);
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) for (var c = 0; c < C; c++) {
      var m = -Infinity;
      for (var dy = 0; dy < 2; dy++) for (var dx = 0; dx < 2; dx++) {
        var v = inp[((2 * y + dy) * W + 2 * x + dx) * C + c]; if (v > m) m = v;
      }
      out[(y * w + x) * C + c] = m;
    }
    return out;
  }
  function XrayNet(X) {
    this.X = X; this.L = X.channels.length - 1; this.size = X.size || 64; this.w = []; this.b = [];
    for (var i = 0; i < this.L; i++) { this.w.push(Float32Array.from(X["w" + i].data)); this.b.push(Float32Array.from(X["b" + i].data)); }
    this.wd = Float32Array.from(X.wd.data); this.bd = X.bd.data[0];
  }
  XrayNet.prototype.run = function (pix) { // pix: size*size values 0..255 (row-major)
    var X = this.X, CH = X.channels, S = this.size, n = S * S, a = new Float32Array(n);
    for (var i = 0; i < n; i++) a[i] = (pix[i] / 255 - X.mean) / X.std;
    for (var l = 0; l < this.L; l++) {
      a = conv3(a, S, S, CH[l], this.w[l], this.b[l], CH[l + 1]);
      if (l < this.L - 1) { a = pool2(a, S, S, CH[l + 1]); S /= 2; }
    }
    var C = CH[this.L], z = this.bd, cam = new Float32Array(S * S);
    for (var p = 0; p < S * S; p++) {
      var s = 0; for (var c = 0; c < C; c++) s += a[p * C + c] * this.wd[c];
      cam[p] = s; z += s / (S * S);
    }
    var prob = 1 / (1 + Math.exp(-z));
    return { prob: prob, flag: prob >= X.threshold, cam: cam, camSize: S };
  };

  // ---------- 3D CNN (CT) ----------
  function conv3d(inp, D, C, w, b, O) { // cube D^3, SAME padding, DHWIO weights, ReLU
    var out = new Float32Array(D * D * D * O);
    for (var z = 0; z < D; z++) for (var y = 0; y < D; y++) for (var x = 0; x < D; x++) {
      var ob = ((z * D + y) * D + x) * O;
      for (var o = 0; o < O; o++) out[ob + o] = b[o];
      for (var kz = 0; kz < 3; kz++) { var zz = z + kz - 1; if (zz < 0 || zz >= D) continue;
        for (var ky = 0; ky < 3; ky++) { var yy = y + ky - 1; if (yy < 0 || yy >= D) continue;
          for (var kx = 0; kx < 3; kx++) { var xx = x + kx - 1; if (xx < 0 || xx >= D) continue;
            var ib = ((zz * D + yy) * D + xx) * C, wb = ((kz * 3 + ky) * 3 + kx) * C * O;
            for (var c = 0; c < C; c++) { var v = inp[ib + c]; if (v === 0) continue; var wo = wb + c * O;
              for (var o2 = 0; o2 < O; o2++) out[ob + o2] += v * w[wo + o2]; } } } }
      for (var o3 = 0; o3 < O; o3++) if (out[ob + o3] < 0) out[ob + o3] = 0;
    }
    return out;
  }
  function pool3d(inp, D, C) {
    var h = D / 2, out = new Float32Array(h * h * h * C);
    for (var z = 0; z < h; z++) for (var y = 0; y < h; y++) for (var x = 0; x < h; x++) for (var c = 0; c < C; c++) {
      var m = -Infinity;
      for (var dz = 0; dz < 2; dz++) for (var dy = 0; dy < 2; dy++) for (var dx = 0; dx < 2; dx++) {
        var v = inp[(((2 * z + dz) * D + 2 * y + dy) * D + 2 * x + dx) * C + c]; if (v > m) m = v; }
      out[((z * h + y) * h + x) * C + c] = m;
    }
    return out;
  }
  function CTNet(X) {
    this.X = X; this.L = X.channels.length - 1; this.w = []; this.b = [];
    for (var i = 0; i < this.L; i++) { this.w.push(Float32Array.from(X["w" + i].data)); this.b.push(Float32Array.from(X["b" + i].data)); }
    this.wd = Float32Array.from(X.wd.data); this.bd = X.bd.data[0];
  }
  CTNet.prototype.run = function (vox) { // vox: 28^3 values 0..255, index (z*28+y)*28+x
    var X = this.X, CH = X.channels, D = X.size, n = D * D * D, a = new Float32Array(n);
    for (var i = 0; i < n; i++) a[i] = (vox[i] / 255 - X.mean) / X.std;
    for (var l = 0; l < this.L; l++) {
      a = conv3d(a, D, CH[l], this.w[l], this.b[l], CH[l + 1]);
      if (l < this.L - 1) { a = pool3d(a, D, CH[l + 1]); D /= 2; }
    }
    var C = CH[this.L], m = D * D * D, z = this.bd, cam = new Float32Array(m);
    for (var p = 0; p < m; p++) { var s = 0; for (var c = 0; c < C; c++) s += a[p * C + c] * this.wd[c]; cam[p] = s; z += s / m; }
    var prob = 1 / (1 + Math.exp(-z));
    return { prob: prob, flag: prob >= X.threshold, cam: cam, camSize: D };
  };

  return { Forecaster: Forecaster, risk: risk, news2: news2, XrayNet: XrayNet, CTNet: CTNet, parseNaive: parseNaive };
})();
if (typeof module !== "undefined") module.exports = FC;
