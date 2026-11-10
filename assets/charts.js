/* Small SVG charts for the project page. Each .chart element holds its data in a child
   <script type="application/json">; on the published page those elements live inside encrypted
   sections, so the charts render after the page is unlocked ('page-unlocked'). No data here. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var W = 640, H = 300, M = { t: 14, r: 14, b: 46, l: 50 };

  function s(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(e);
    return e;
  }

  function niceStep(v) {
    var p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
  }

  // Axis maximum and tick count on a 1-2-2.5-5 step, about four ticks.
  function niceAxis(v) {
    var step = niceStep(v / 4);
    return { max: Math.ceil(v / step - 1e-9) * step, ticks: Math.round(Math.ceil(v / step - 1e-9)) };
  }

  function niceMax(v) { return niceAxis(v).max; }

  function fmt(v, unit) {
    if (v === null || v === undefined) return '–';
    var d = Math.abs(v) < 10 ? 2 : 1;
    return v.toFixed(d) + (unit ? ' ' + unit : '');
  }

  function tooltip(box) {
    var t = document.createElement('div');
    t.className = 'chart-tip';
    t.hidden = true;
    box.appendChild(t);
    return {
      show: function (html, x, y) {
        t.innerHTML = html;
        t.hidden = false;
        var r = box.getBoundingClientRect(), tw = t.offsetWidth;
        var left = Math.min(Math.max(x - r.left + 12, 4), r.width - tw - 4);
        t.style.left = left + 'px';
        t.style.top = Math.max(y - r.top - t.offsetHeight - 10, 4) + 'px';
      },
      hide: function () { t.hidden = true; }
    };
  }

  function legend(box, series) {
    var lg = document.createElement('div');
    lg.className = 'chart-legend';
    series.forEach(function (se) {
      var sp = document.createElement('span');
      sp.innerHTML = '<i class="sw ' + (se.dash ? 'dash' : '') + '" style="--c:var(--c-' + se.key + ')"></i>' + se.name;
      lg.appendChild(sp);
    });
    box.appendChild(lg);
  }

  function axes(svg, yMax, unit, ticks) {
    var g = s('g', { class: 'axis' }, svg);
    for (var i = 0; i <= ticks; i++) {
      var v = yMax * i / ticks, y = M.t + (H - M.t - M.b) * (1 - i / ticks);
      s('line', { x1: M.l, x2: W - M.r, y1: y, y2: y, class: i ? 'grid' : 'base' }, g);
      var tx = s('text', { x: M.l - 8, y: y + 4, 'text-anchor': 'end' }, g);
      tx.textContent = +v.toFixed(2) + (i === ticks && unit ? ' ' + unit : '');
    }
  }

  function yScale(yMax) { return function (v) { return M.t + (H - M.t - M.b) * (1 - v / yMax); }; }

  // ---------------------------------------------------------------- grouped vertical bars
  function bars(svg, d, tip, unit) {
    var all = [];
    d.series.forEach(function (se) { se.values.forEach(function (v) { if (v !== null) all.push(v); }); });
    var ax = niceAxis(d.yMax || Math.max.apply(null, all) * 1.03), yMax = ax.max, y = yScale(yMax);
    axes(svg, yMax, unit, ax.ticks);
    var n = d.categories.length, k = d.series.length, cw = (W - M.l - M.r) / n;
    var bw = Math.min(26, (cw * 0.78) / k), gap = 2;
    d.categories.forEach(function (c, i) {
      var x0 = M.l + cw * i + (cw - (bw * k + gap * (k - 1))) / 2;
      d.series.forEach(function (se, j) {
        var v = se.values[i];
        if (v === null) return;
        var x = x0 + j * (bw + gap), top = y(Math.min(v, yMax)), h = y(0) - top;
        var r = Math.min(4, bw / 2, h);
        var path = 'M' + x + ',' + y(0) + 'V' + (top + r) + 'Q' + x + ',' + top + ' ' + (x + r) + ',' + top +
          'H' + (x + bw - r) + 'Q' + (x + bw) + ',' + top + ' ' + (x + bw) + ',' + (top + r) + 'V' + y(0) + 'Z';
        s('path', { d: path, class: 'bar', style: 'fill:var(--c-' + se.key + ')' }, svg);
        var hit = s('rect', { x: x - gap / 2, y: M.t, width: bw + gap, height: y(0) - M.t, class: 'hit' }, svg);
        hit.addEventListener('mousemove', function (ev) {
          tip.show('<b>' + c + '</b><br><i class="sw" style="--c:var(--c-' + se.key + ')"></i>' + se.name + ': <b>' +
            fmt(v, unit) + '</b>' + (d.notes && d.notes[i] ? '<br><span class="muted">' + d.notes[i] + '</span>' : ''),
          ev.clientX, ev.clientY);
        });
        hit.addEventListener('mouseleave', tip.hide);
      });
      var t = s('text', { x: M.l + cw * (i + 0.5), y: H - M.b + 18, 'text-anchor': 'middle', class: 'cat' }, svg);
      t.textContent = c;
      if (d.sub && d.sub[i]) {
        var t2 = s('text', { x: M.l + cw * (i + 0.5), y: H - M.b + 33, 'text-anchor': 'middle', class: 'sub' }, svg);
        t2.textContent = d.sub[i];
      }
    });
  }

  // ---------------------------------------------------------------- lines over a numeric x
  function lines(svg, d, tip, unit, box) {
    var all = [];
    d.series.forEach(function (se) { se.values.forEach(function (v) { if (v !== null) all.push(v); }); });
    var ax = niceAxis(d.yMax || Math.max.apply(null, all) * 1.03), yMax = ax.max, y = yScale(yMax);
    axes(svg, yMax, unit, ax.ticks);
    var n = d.x.length, x = function (i) { return M.l + 18 + (W - M.l - M.r - 36) * i / (n - 1); };
    d.x.forEach(function (lab, i) {
      var t = s('text', { x: x(i), y: H - M.b + 18, 'text-anchor': 'middle', class: 'cat' }, svg);
      t.textContent = lab;
    });
    if (d.xLabel) {
      var xl = s('text', { x: (M.l + W - M.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'sub' }, svg);
      xl.textContent = d.xLabel;
    }
    d.series.forEach(function (se) {
      var pts = [];
      se.values.forEach(function (v, i) { if (v !== null) pts.push([x(i), y(v)]); });
      s('polyline', { points: pts.map(function (p) { return p.join(','); }).join(' '), class: 'line' + (se.dash ? ' dash' : ''),
        style: 'stroke:var(--c-' + se.key + ')' }, svg);
      pts.forEach(function (p) { s('circle', { cx: p[0], cy: p[1], r: 4, class: 'dot', style: 'fill:var(--c-' + se.key + ')' }, svg); });
    });
    var cross = s('line', { y1: M.t, y2: H - M.b, class: 'cross', visibility: 'hidden' }, svg);
    var hit = s('rect', { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, class: 'hit' }, svg);
    hit.addEventListener('mousemove', function (ev) {
      var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
      var i = Math.max(0, Math.min(n - 1, Math.round((px - M.l - 18) / ((W - M.l - M.r - 36) / (n - 1)))));
      cross.setAttribute('x1', x(i));
      cross.setAttribute('x2', x(i));
      cross.setAttribute('visibility', 'visible');
      var rows = d.series.map(function (se) {
        var extra = se.extra ? ' <span class="muted">' + se.extra[i] + '</span>' : '';
        return '<i class="sw" style="--c:var(--c-' + se.key + ')"></i>' + se.name + ': <b>' + fmt(se.values[i], unit) + '</b>' + extra;
      });
      tip.show('<b>' + (d.xTip || '') + d.x[i] + '</b><br>' + rows.join('<br>'), ev.clientX, ev.clientY);
    });
    hit.addEventListener('mouseleave', function () { tip.hide(); cross.setAttribute('visibility', 'hidden'); });
    void box;
  }

  // ---------------------------------------------------------------- single-series horizontal bars
  function hbars(svg, d, tip, unit) {
    var n = d.labels.length, rowH = 30, h = n * rowH + 26, lw = Math.min(210, W * 0.42);
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + h);
    var xMax = d.xMax || niceMax(Math.max.apply(null, d.values) * 1.08);
    var x = function (v) { return lw + (W - lw - 60) * v / xMax; };
    s('line', { x1: lw, x2: lw, y1: 4, y2: n * rowH + 4, class: 'base' }, svg);
    d.labels.forEach(function (lab, i) {
      var y0 = 4 + i * rowH, v = d.values[i], key = d.keys ? d.keys[i] : d.key;
      var t = s('text', { x: lw - 10, y: y0 + rowH / 2 + 4, 'text-anchor': 'end', class: 'cat' }, svg);
      t.textContent = lab;
      var bw = x(v) - lw, r = Math.min(4, bw);
      s('path', { d: 'M' + lw + ',' + (y0 + 6) + 'H' + (lw + bw - r) + 'Q' + (lw + bw) + ',' + (y0 + 6) + ' ' + (lw + bw) + ',' + (y0 + 6 + r) +
        'V' + (y0 + rowH - 6 - r) + 'Q' + (lw + bw) + ',' + (y0 + rowH - 6) + ' ' + (lw + bw - r) + ',' + (y0 + rowH - 6) + 'H' + lw + 'Z',
        class: 'bar', style: 'fill:var(--c-' + key + ')' }, svg);
      var vt = s('text', { x: lw + bw + 6, y: y0 + rowH / 2 + 4, class: 'val' }, svg);
      vt.textContent = fmt(v, unit);
      var hit = s('rect', { x: 0, y: y0, width: W, height: rowH, class: 'hit' }, svg);
      hit.addEventListener('mousemove', function (ev) {
        tip.show('<b>' + lab + '</b><br>' + fmt(v, unit) + (d.notes && d.notes[i] ? '<br><span class="muted">' + d.notes[i] + '</span>' : ''), ev.clientX, ev.clientY);
      });
      hit.addEventListener('mouseleave', tip.hide);
    });
  }

  var KINDS = { bars: bars, lines: lines, hbars: hbars };

  function render(el) {
    if (el.dataset.ready) return;
    var src = el.querySelector('script[type="application/json"]');
    if (!src) return;
    el.dataset.ready = '1';
    var spec = JSON.parse(src.textContent), variants = spec.variants || [spec], cur = 0;
    var bar = document.createElement('div');
    bar.className = 'chart-switch';
    var box = document.createElement('div');
    box.className = 'chart-box';
    if (variants.length > 1) el.appendChild(bar);
    el.appendChild(box);
    var tip = tooltip(el);
    function draw() {
      box.innerHTML = '';
      W = Math.max(300, Math.round(box.clientWidth || 640));
      H = W < 520 ? 250 : 300;
      var d = variants[cur], svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': spec.title || '' }, box);
      KINDS[spec.kind](svg, d, tip, spec.unit, box);
      if (d.series) legend(box, d.series);
    }
    variants.forEach(function (v, i) {
      if (variants.length < 2) return;
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = v.label;
      b.setAttribute('aria-pressed', i === cur);
      b.addEventListener('click', function () {
        cur = i;
        bar.querySelectorAll('button').forEach(function (x, j) { x.setAttribute('aria-pressed', j === cur); });
        draw();
      });
      bar.appendChild(b);
    });
    draw();
    redraws.push(draw);
  }

  var redraws = [], timer = null;
  window.addEventListener('resize', function () {
    clearTimeout(timer);
    timer = setTimeout(function () { redraws.forEach(function (f) { f(); }); }, 150);
  });

  function boot() { document.querySelectorAll('.chart').forEach(render); }
  window.addEventListener('page-unlocked', boot);
  window.addEventListener('DOMContentLoaded', boot);
})();
