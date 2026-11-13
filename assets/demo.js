/* Interactive demo: clip video with YOLO / PnLCalib / trajectory overlays and a three.js 3D view.
   Local mode reads assets/demo/*; on the published page the assets are AES-GCM encrypted and are
   decrypted with the key derived when the visitor unlocks the page (window.__pageVault). */
(function () {
  'use strict';
  // One solid colour per method, the same in the video overlay and the 3D view.
  var COLORS = { 'Ground truth': '#ffffff', 'ArcPrior Base': '#E69F00', 'ArcPrior ft': '#D55E00',
    'Grad et al.': '#56B4E9', 'LM-fitg': '#009E73' };
  // The data's world frame (x, y, z up) is left-handed, so the 3D view mirrors y to show the pitch
  // as the camera sees it.
  var MIRROR_Y = -1;
  // Consecutive points further apart than this (75 m/s at 25 fps) are drawn as a gap: the benchmark's
  // ground truth has a few outlier frames.
  var MAX_STEP_M = 3;
  var BALL_RADIUS_M = 0.11;   // a size-5 ball is 22 cm across

  // Split a path at missing frames and implausible jumps; each piece has at least two points.
  function pieces(path) {
    var out = [], run = [];
    path.forEach(function (q) {
      var last = run[run.length - 1];
      if (!q || (last && Math.hypot(q[0] - last[0], q[1] - last[1], q[2] - last[2]) > MAX_STEP_M)) {
        if (run.length > 1) out.push(run);
        run = q ? [q] : [];
      } else run.push(q);
    });
    if (run.length > 1) out.push(run);
    return out;
  }

  async function getAsset(name, kind) {
    var vault = window.__pageVault;
    if (vault) {
      var path = vault.assets[name];
      if (!path) throw new Error('missing asset ' + name);
      var buf = new Uint8Array(await (await fetch(path)).arrayBuffer());
      var plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: buf.slice(0, 12) }, vault.key, buf.slice(12));
      if (kind === 'json') return JSON.parse(new TextDecoder().decode(plain));
      return URL.createObjectURL(new Blob([plain], { type: 'video/mp4' }));
    }
    var res = await fetch('assets/demo/' + name);
    if (kind === 'json') return res.json();
    return URL.createObjectURL(await res.blob());
  }

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text) e.textContent = text;
    return e;
  }

  // ---------------------------------------------------------------- pitch geometry (metres)
  function pitchLines(p) {
    var cx = p.centre[0], cy = p.centre[1], L = p.length / 2, W = p.width / 2, out = [];
    function rect(x0, y0, x1, y1) { out.push([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]); }
    function arc(x, y, r, a0, a1) { var pts = []; for (var i = 0; i <= 40; i++) { var a = a0 + (a1 - a0) * i / 40; pts.push([x + r * Math.cos(a), y + r * Math.sin(a)]); } out.push(pts); }
    rect(cx - L, cy - W, cx + L, cy + W);
    out.push([[cx, cy - W], [cx, cy + W]]);
    arc(cx, cy, 9.15, 0, 2 * Math.PI);
    [-1, 1].forEach(function (s) {
      var gx = cx + s * L;
      rect(Math.min(gx, gx - s * 16.5), cy - 20.16, Math.max(gx, gx - s * 16.5), cy + 20.16);
      rect(Math.min(gx, gx - s * 5.5), cy - 9.16, Math.max(gx, gx - s * 5.5), cy + 9.16);
      var px = gx - s * 11, a = Math.acos(5.5 / 9.15);
      if (s > 0) arc(px, cy, 9.15, Math.PI - a, Math.PI + a); else arc(px, cy, 9.15, -a, a);
    });
    return out;
  }

  function project(P, x, y, z) {
    var u = P[0] * x + P[1] * y + P[2] * z + P[3], v = P[4] * x + P[5] * y + P[6] * z + P[7],
        w = P[8] * x + P[9] * y + P[10] * z + P[11];
    if (w <= 1e-6) return null;
    return [u / w, v / w];
  }

  // ---------------------------------------------------------------- 3D view
  function make3d(container) {
    if (!window.THREE) { container.textContent = '3D view needs three.js (offline?)'; return null; }
    var renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    container.appendChild(renderer.domElement);
    var scene = new THREE.Scene();
    scene.background = new THREE.Color('#0f1f14');
    var camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.5, 1000);
    camera.up.set(0, 0, 1);
    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    var group = new THREE.Group();
    group.scale.set(1, MIRROR_Y, 1);
    scene.add(group);
    var state = { follow: true };
    controls.addEventListener('start', function () { state.follow = false; });
    function resize() {
      var w = container.clientWidth, h = container.clientHeight || w * 9 / 16;
      renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);
    resize();
    (function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
    return { scene: scene, camera: camera, controls: controls, group: group, resize: resize, state: state };
  }

  // Place the 3D camera at the video camera's pose: P = K [R | t] on the 960 x 540 overlay.
  function followCamera(v, P, target) {
    var m = [P.slice(0, 3), P.slice(4, 7), P.slice(8, 11)], p4 = [P[3], P[7], P[11]];
    function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    function axpy(a, s, b) { return [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]; }
    var k33 = Math.hypot.apply(null, m[2]), r3 = m[2].map(function (x) { return x / k33; });
    var u = axpy(m[1], -dot(m[1], r3), r3), fy = Math.hypot.apply(null, u) / k33;
    // camera centre C solves M C = -p4 (Cramer's rule)
    function det(a) {
      return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
        a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    }
    var d = det(m), C = [0, 1, 2].map(function (i) {
      return det(m.map(function (row, j) { return row.map(function (x, k) { return k === i ? -p4[j] : x; }); })) / d;
    });
    var depth = Math.max(5, dot([target[0] - C[0], target[1] - C[1], target[2] - C[2]], r3));
    var look = axpy(C, depth, r3);
    v.camera.fov = 2 * Math.atan(270 / fy) * 180 / Math.PI;
    v.camera.updateProjectionMatrix();
    v.camera.position.set(C[0], MIRROR_Y * C[1], C[2]);
    v.controls.target.set(look[0], MIRROR_Y * look[1], look[2]);
    v.controls.update();
  }

  function line3(points, color) {
    var g = new THREE.BufferGeometry().setFromPoints(points.map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); }));
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color: color }));
  }

  // Solid tubes along a path, one per piece.
  function tubes(path, color, radius, flat) {
    return pieces(path).map(function (run) {
      var curve = new THREE.CatmullRomCurve3(run.map(function (q) { return new THREE.Vector3(q[0], q[1], flat ? 0.03 : q[2]); }));
      return new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, run.length * 3), radius, 8, false),
        new THREE.MeshBasicMaterial({ color: color }));
    });
  }

  function draw3d(v, clip, shown, opts) {
    if (!v) return;
    while (v.group.children.length) v.group.remove(v.group.children[0]);
    var p = clip.pitch;
    var plane = new THREE.Mesh(new THREE.PlaneGeometry(p.length + 10, p.width + 10),
      new THREE.MeshBasicMaterial({ color: '#2f6b3a', side: THREE.DoubleSide }));
    plane.position.set(p.centre[0], p.centre[1], -0.02);
    v.group.add(plane);
    pitchLines(p).forEach(function (pl) { v.group.add(line3(pl.map(function (q) { return [q[0], q[1], 0.01]; }), '#e8f5e9')); });
    shown.forEach(function (s) {
      var col = COLORS[s.name] || '#ffffff';
      if (opts.air) tubes(s.path, col, 0.08, false).forEach(function (m) { v.group.add(m); });
      if (opts.ground) tubes(s.path, col, 0.05, true).forEach(function (m) { v.group.add(m); });
    });
    var marker = new THREE.Mesh(new THREE.SphereGeometry(BALL_RADIUS_M, 24, 16), new THREE.MeshBasicMaterial({ color: '#ff3b30' }));
    v.group.add(marker);
    v.marker = marker;
  }

  function centroid(clip) {
    var pts = clip.gt.filter(function (q) { return q; });
    if (!pts.length) return [clip.pitch.centre[0], clip.pitch.centre[1], 0];
    return pts.reduce(function (a, q) { return [a[0] + q[0], a[1] + q[1], a[2] + q[2]]; }, [0, 0, 0]).map(function (x) { return x / pts.length; });
  }

  // ---------------------------------------------------------------- app
  var SETTINGS = [['gt', 'GT inputs: dataset track and camera'], ['real', 'Real: published YOLO + PnLCalib'],
    ['heads', 'Real + heads ft (det,cal)']];
  var PRIORS = [['frozen', 'Prior: chosen on validation'], ['none', 'Prior: none'], ['gauss', 'Prior: Gaussian'],
    ['subspace1', 'Prior: weak-direction, weight 1'], ['subspace3', 'Prior: weak-direction, weight 3']];
  // Display name -> method key in the clip file ('base' / 'ft' are followed by the prior).
  var METHODS = [['LM-fitg', 'lm_fitg'], ['Grad et al.', 'grad'], ['ArcPrior Base', 'base'], ['ArcPrior ft', 'ft']];

  function ordinal(q) { return q + (q === 100 ? 'th' : ['th', 'st', 'nd', 'rd'][q % 10] || 'th'); }

  async function start(root) {
    root.innerHTML = '';
    var manifest = await getAsset('manifest.json', 'json');
    var flights = {};
    manifest.forEach(function (m) { (flights[m.flight] = flights[m.flight] || []).push(m); });
    var controls = el('div', { class: 'demo-controls' });
    function select(label, options) {
      var s = el('select', { 'aria-label': label });
      options.forEach(function (o) { s.appendChild(el('option', { value: o[0] }, o[1])); });
      return s;
    }
    var selFlight = select('Flight', Object.keys(flights).sort(function (x, y) {
      return flights[x][0].percentile - flights[y][0].percentile;
    }).map(function (f) {
      var m = flights[f][0];
      return [f, ordinal(m.percentile) + ' percentile (half ' + m.half + ', apex ' + m.apex + ' m)'];
    }));
    var selView = el('select', { 'aria-label': 'Camera' });
    var selSetting = select('Inputs', SETTINGS);
    var levels = [['true', 'correct'], ['off', 'wrong (0.5 m off)'], ['none', 'not given']];
    var selStart = select('Start contact height', levels.map(function (l) { return [l[0], 'Start height: ' + l[1]]; }));
    var selEnd = select('End contact height', levels.map(function (l) { return [l[0], 'End height: ' + l[1]]; }));
    var selPrior = select('ArcPrior prior', PRIORS);
    var selShare = select('Seen part of the flight', [['1', 'Whole flight seen (end contact known)']].concat(
      [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3].map(function (q) { return [String(q), Math.round(q * 100) + ' % seen, rest forecast']; })));
    var toggles = {};
    function toggle(key, label, on) {
      var l = el('label'), c = el('input', { type: 'checkbox' });
      c.checked = on; l.appendChild(c); l.appendChild(document.createTextNode(label));
      toggles[key] = c; return l;
    }
    var playBtn = el('button', { type: 'button' }, 'Pause');
    [selFlight, selView, selSetting, selShare, selStart, selEnd, selPrior].forEach(function (x) { controls.appendChild(x); });
    var toggleBox = el('div', { class: 'demo-controls' });
    [toggle('yolo', 'YOLO boxes', true), toggle('pnl', 'Pitch lines', true), toggle('track', 'Tracked ball', true),
     toggle('air', '3D paths', true), toggle('ground', 'Ground tracks', false), playBtn].forEach(function (x) { toggleBox.appendChild(x); });
    var methodBox = el('div', { class: 'demo-controls' });
    var grid = el('div', { class: 'demo-grid' });
    var left = el('div'), stage = el('div', { class: 'stage' });
    var video = el('video', { muted: '', playsinline: '', loop: '' }); video.muted = true;
    var canvas = el('canvas', { width: 960, height: 540 });
    stage.appendChild(video); stage.appendChild(canvas);
    var scrub = el('input', { type: 'range', min: 0, max: 100, value: 0, class: 'scrub', 'aria-label': 'Frame' });
    var status = el('p', { class: 'status' });
    var info = el('p', { class: 'status' });
    left.appendChild(stage); left.appendChild(scrub); left.appendChild(status); left.appendChild(info);
    var right = el('div', { class: 'view3d' });
    var hint = el('p', { class: 'status' }, '3D view: starts at the video camera and follows it; drag to rotate, scroll to zoom. Changing the flight or camera resets it.');
    grid.appendChild(left); grid.appendChild(right); grid.appendChild(hint);
    var legend = el('div', { class: 'legend' });
    root.appendChild(controls); root.appendChild(toggleBox); root.appendChild(methodBox); root.appendChild(grid); root.appendChild(legend);
    var view = make3d(right);
    var ctx = canvas.getContext('2d');
    var clip = null, methodChecks = {}, cache = {};

    function pairIndex() { return clip.levels.indexOf(selStart.value) * 3 + clip.levels.indexOf(selEnd.value); }
    function isPartial() { return selShare.value !== '1'; }
    function partIndex() { return clip.shares.indexOf(Number(selShare.value)) * 3 + clip.levels.indexOf(selStart.value); }
    // Frames seen at the current share (the whole flight when it is not cut).
    function seenFrames() {
      if (!isPartial()) return clip.flight_n;
      var e = entry('base');
      return e && e.cut ? e.cut : Math.round(Number(selShare.value) * clip.flight_n);
    }

    // The entry of one method for the current inputs, heights and prior (null if not run or not answered).
    function entry(key) {
      var setting = selSetting.value, ms = clip.methods[setting] || {};
      if (key === 'base' || key === 'ft') {
        var prior = selPrior.value === 'frozen' ? clip.frozen[setting + ':' + key] : selPrior.value;
        if (!prior) return null;
        key = key + ':' + prior;
      }
      if (isPartial()) {
        var part = (clip.partial[setting] || {})[key];
        return part ? part[partIndex()] : null;
      }
      var list = ms[key];
      return list ? list[pairIndex()] : null;
    }

    function shownPaths() {
      var out = [];
      if (!clip) return out;
      if (methodChecks['Ground truth'] && methodChecks['Ground truth'].checked) out.push({ name: 'Ground truth', path: clip.gt });
      METHODS.forEach(function (m) {
        var e = entry(m[1]);
        if (e && methodChecks[m[0]] && methodChecks[m[0]].checked) out.push({ name: m[0], path: e.path, rmse: e.rmse });
      });
      return out;
    }

    function buildMethods() {
      methodBox.innerHTML = ''; legend.innerHTML = '';
      var names = ['Ground truth'].concat(METHODS.map(function (m) { return m[0]; }));
      names.forEach(function (n) {
        var key = (METHODS.filter(function (m) { return m[0] === n; })[0] || [])[1];
        var e = key ? entry(key) : null;
        if (key === 'ft' && selSetting.value === 'gt') return;
        if (key === 'grad' && isPartial()) return;   // Grad et al. fit only the frames seen
        var prev = methodChecks[n] ? methodChecks[n].checked : true;
        var l = el('label'), c = el('input', { type: 'checkbox' });
        c.checked = prev; c.addEventListener('change', refresh);
        methodChecks[n] = c;
        l.appendChild(c);
        var text = n;
        if (key && isPartial()) text += e && e.coverage > 0 ? ' (RMSE ' + e.rmse.toFixed(2) + ' m; forecast ' + e.rmse_unseen.toFixed(2) + ' m)' : ' (no answer)';
        else if (key) text += e && e.coverage > 0 ? ' (RMSE ' + e.rmse.toFixed(2) + ' m)' : ' (no answer)';
        l.appendChild(document.createTextNode(text));
        methodBox.appendChild(l);
        var s = el('span'), sw = el('i', { class: 'swatch' });
        sw.style.background = COLORS[n]; sw.style.outline = '1px solid #8a8f96'; s.appendChild(sw); s.appendChild(document.createTextNode(n));
        legend.appendChild(s);
      });
      var inp = clip.inputs[selSetting.value], k = isPartial() ? clip.levels.indexOf(selStart.value) * 3 : pairIndex();
      function h(end) {
        var v = inp.heights[end][k];
        return v == null ? 'not given' : v.toFixed(2) + ' m given';
      }
      selEnd.disabled = isPartial();
      var endText = isPartial() ? 'not used, the end is not seen' : h('end');
      var seenText = isPartial() ? ' Seen: the first ' + seenFrames() + ' of ' + clip.flight_n + ' frames; the rest is forecast (the tracked ball is shown on seen frames only).' : '';
      info.textContent = 'Contact heights: start ' + h('start') + ' (true ' + clip.z_contact[0].toFixed(2) + ' m), end ' + endText +
        ' (true ' + clip.z_contact[1].toFixed(2) + ' m). RMSE over the whole flight, capped at 10 m; unanswered frames count as 10 m.' + seenText;
    }

    function frameIndex() { return Math.min(clip.count - 1, Math.max(0, Math.floor(video.currentTime * clip.fps + 1e-3))); }

    function polyline(pts) { ctx.beginPath(); pts.forEach(function (q, i) { if (i) ctx.lineTo(q[0], q[1]); else ctx.moveTo(q[0], q[1]); }); ctx.stroke(); }

    function drawOverlay() {
      if (!clip) return;
      var k = frameIndex(), frame = clip.start + k, fi = frame - clip.flight_start;
      var inp = clip.inputs[selSetting.value], P = clip.camera[k];
      ctx.clearRect(0, 0, 960, 540);
      if (toggles.pnl.checked) {
        ctx.strokeStyle = 'rgba(0,229,255,0.85)'; ctx.lineWidth = 1.6;
        if (inp.lines) {
          (inp.lines.static ? inp.lines.lines : (inp.lines.lines[k] || [])).forEach(polyline);
        } else {
          // GT inputs: the dataset camera's pitch lines.
          pitchLines(clip.pitch).forEach(function (pl) {
            var pts = pl.map(function (q) { return project(P, q[0], q[1], 0); });
            if (pts.every(function (q) { return q; })) polyline(pts);
          });
        }
      }
      if (toggles.yolo.checked && inp.boxes) {
        (inp.boxes[k] || []).forEach(function (b, i) {
          ctx.strokeStyle = i === 0 ? '#ff7f00' : 'rgba(255,255,255,0.8)'; ctx.lineWidth = i === 0 ? 2 : 1;
          ctx.strokeRect(b[0] - 4, b[1] - 4, b[2] - b[0] + 8, b[3] - b[1] + 8);
          ctx.fillStyle = ctx.strokeStyle; ctx.font = '12px Inter, sans-serif';
          ctx.fillText(b[4].toFixed(2), b[2] + 6, b[1] + 2);
        });
      }
      if (toggles.track.checked && fi >= 0 && fi < seenFrames() && inp.track[fi]) {
        var t = inp.track[fi];
        ctx.strokeStyle = '#ffeb3b'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(t[0], t[1], 9, 0, 2 * Math.PI); ctx.stroke();
      }
      if (toggles.air.checked || toggles.ground.checked) {
        shownPaths().forEach(function (s) {
          var col = COLORS[s.name] || '#fff';
          function stroke(z0) {
            ctx.beginPath();
            pieces(s.path).forEach(function (run) {
              var started = false;
              run.forEach(function (q) {
                var uv = project(P, q[0], q[1], z0 ? 0 : q[2]);
                if (!uv) { started = false; return; }
                if (started) ctx.lineTo(uv[0], uv[1]); else { ctx.moveTo(uv[0], uv[1]); started = true; }
              });
            });
            ctx.stroke();
          }
          ctx.strokeStyle = col; ctx.lineWidth = 1.5;
          if (toggles.air.checked) stroke(false);
          if (toggles.ground.checked) stroke(true);
          if (toggles.air.checked && fi >= 0 && fi < s.path.length && s.path[fi]) {
            var b = project(P, s.path[fi][0], s.path[fi][1], s.path[fi][2]);
            if (b) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(b[0], b[1], 4, 0, 2 * Math.PI); ctx.fill(); }
          }
          // Partial flights: a white ring where the seen part ends and the forecast begins.
          var last = s.path[seenFrames() - 1];
          if (isPartial() && toggles.air.checked && last) {
            var c = project(P, last[0], last[1], last[2]);
            if (c) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(c[0], c[1], 6, 0, 2 * Math.PI); ctx.stroke(); }
          }
        });
      }
      if (view && view.state.follow) followCamera(view, P, centroid(clip));
      if (view && view.marker) {
        var gt = clip.gt[Math.min(clip.flight_n - 1, Math.max(0, fi))];
        view.marker.visible = !!gt && fi >= 0 && fi < clip.flight_n;
        if (gt) view.marker.position.set(gt[0], gt[1], gt[2]);
      }
      scrub.value = Math.round(100 * k / Math.max(1, clip.count - 1));
      status.textContent = clip.view + ', frame ' + frame + (fi >= 0 && fi < clip.flight_n ? (fi < seenFrames() ? ' (in flight, seen)' : ' (in flight, forecast)') : '');
    }

    function refresh() {
      if (!clip) return;
      draw3d(view, clip, shownPaths(), { air: toggles.air.checked, ground: toggles.ground.checked });
      drawOverlay();
    }

    function rebuild() {
      if (!clip) return;
      // Clips without partial-flight data offer the whole flight only.
      selShare.style.display = clip.shares ? '' : 'none';
      if (!clip.shares) selShare.value = '1';
      buildMethods(); refresh();
    }

    async function loadClip() {
      var f = selFlight.value, v = selView.value || flights[f][0].view, name = f + '_' + v;
      status.textContent = 'Loading ' + name + '…';
      if (!cache[name]) cache[name] = { data: await getAsset(name + '.json', 'json'), url: await getAsset(name + '.mp4', 'video') };
      clip = cache[name].data;
      if (view) view.state.follow = true;
      video.src = cache[name].url; video.play().catch(function () {});
      rebuild();
    }

    function fillViews() {
      selView.innerHTML = '';
      flights[selFlight.value].forEach(function (m) { selView.appendChild(el('option', { value: m.view }, m.view)); });
    }
    selFlight.addEventListener('change', function () { fillViews(); loadClip(); });
    selView.addEventListener('change', loadClip);
    [selSetting, selShare, selStart, selEnd, selPrior].forEach(function (s) { s.addEventListener('change', rebuild); });
    ['yolo', 'pnl', 'track'].forEach(function (k) { toggles[k].addEventListener('change', drawOverlay); });
    ['air', 'ground'].forEach(function (k) { toggles[k].addEventListener('change', refresh); });
    playBtn.addEventListener('click', function () { if (video.paused) { video.play(); playBtn.textContent = 'Pause'; } else { video.pause(); playBtn.textContent = 'Play'; } });
    scrub.addEventListener('input', function () { video.pause(); playBtn.textContent = 'Play'; video.currentTime = (scrub.value / 100) * (clip.count - 1) / clip.fps; });
    (function tick() { drawOverlay(); requestAnimationFrame(tick); })();
    fillViews();
    await loadClip();
    if (view) view.resize();
  }

  function boot() {
    var root = document.getElementById('demo-app');
    if (!root || root.dataset.started) return;
    root.dataset.started = '1';
    start(root).catch(function (e) { root.textContent = 'Demo failed to load (' + e.message + '). Reload the page; if it persists, clear the browser cache for this site.'; });
  }
  window.addEventListener('page-unlocked', boot);
  window.addEventListener('DOMContentLoaded', function () { if (!document.getElementById('vault')) boot(); });
})();
