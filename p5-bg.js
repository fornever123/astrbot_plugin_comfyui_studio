/* ==========================================================
   p5-bg.js · 常驻动态背景（Canvas 2D）
   放射光束缓慢旋转 · 碎片漂移 + 鼠标视差 · 星星闪烁 · 偶尔一道斜光
   简化动效 / 页面隐藏时停帧（只画一张静态图）；ComfyUI 断线时转灰并抖动。
   帧率上限 30fps，背景只是氛围，不跟操作抢资源。
   ========================================================== */
(() => {
  "use strict";
  const cv = document.getElementById("p5Bg");
  if (!cv || !cv.getContext) return;
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  const root = document.documentElement, R = Math.random;
  let W = 0, H = 0, dpr = 1, shards = [], stars = [], streak = null, nextStreak = 2500, last = 0, acc = 0, t = 0, raf = 0;
  let mx = 0, my = 0, tmx = 0, tmy = 0, glitchT = 0;
  const full = () => root.dataset.motion !== "lite";
  const off = () => root.classList.contains("p5-off");
  const dark = () => root.dataset.uiTheme === "dark";

  function mkShard(init) {
    const n = 3 + ((R() * 3) | 0), size = 26 + Math.pow(R(), 2.2) * 260, pts = [];
    for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2 + R() * .9, rr = size * (.4 + R() * .6); pts.push([Math.cos(a) * rr, Math.sin(a) * rr]); }
    const kind = R() < .58 ? 0 : R() < .55 ? 1 : 2;
    return { x: init ? R() * (W + 300) - 150 : W + size + R() * 120, y: R() * (H + 200) - 100, pts, size, rot: R() * 6.28, vr: (R() - .5) * .35, vx: -(10 + R() * 28) * (size < 90 ? 1.5 : 1), vy: (R() - .35) * 10, depth: .25 + R() * .9, kind };
  }
  function init() {
    dpr = Math.min(1.5, window.devicePixelRatio || 1); W = innerWidth; H = innerHeight;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.min(30, 12 + (W * H) / 80000));
    shards = Array.from({ length: n }, () => mkShard(true)).sort((a, b) => a.depth - b.depth);
    stars = Array.from({ length: 40 }, () => ({ x: R() * W, y: R() * H, s: 3 + R() * 9, ph: R() * 6.28, sp: .8 + R() * 2.2, vx: -(4 + R() * 12) }));
  }
  function starPath(x, y, r, rot) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) { const a = rot + (i * Math.PI) / 5 - Math.PI / 2, rr = i % 2 ? r * .42 : r; ctx[i ? "lineTo" : "moveTo"](x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
    ctx.closePath();
  }
  function draw(dt) {
    const D = dark(), O = off(), F = full(), s = dt / 1000;
    const g = ctx.createLinearGradient(0, 0, W * .7, H);
    if (D) { g.addColorStop(0, "#141414"); g.addColorStop(1, "#050505"); } else { g.addColorStop(0, "#D9001C"); g.addColorStop(.55, "#BA0018"); g.addColorStop(1, "#7E0010"); }
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    // 放射光束
    const cx = W * .8 + mx * -30, cy = H * .22 + my * -20, rays = 18, rot = O ? 0 : t * .00006, Rr = Math.hypot(W, H);
    ctx.fillStyle = D ? "rgba(229,0,26,.075)" : "rgba(0,0,0,.11)";
    ctx.beginPath();
    for (let i = 0; i < rays; i++) { const a = rot + (i / rays) * Math.PI * 2, w = Math.PI / rays * .55; ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a - w) * Rr, cy + Math.sin(a - w) * Rr); ctx.lineTo(cx + Math.cos(a + w) * Rr, cy + Math.sin(a + w) * Rr); ctx.closePath(); }
    ctx.fill();
    // 同心环
    ctx.strokeStyle = D ? "rgba(229,0,26,.12)" : "rgba(0,0,0,.14)"; ctx.lineWidth = 2;
    for (let k = 0; k < 3; k++) { const rr = ((t * .02 + k * 160) % 480) + 60; ctx.globalAlpha = 1 - rr / 540; ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.stroke(); }
    ctx.globalAlpha = 1;
    // 碎片
    for (const p of shards) {
      if (F && !O) { p.x += p.vx * s; p.y += p.vy * s; p.rot += p.vr * s; }
      if (p.x < -p.size - 60) Object.assign(p, mkShard(false));
      if (p.y < -p.size - 80) p.y = H + p.size;
      if (p.y > H + p.size + 80) p.y = -p.size;
      let ox = mx * -60 * p.depth, oy = my * -40 * p.depth;
      if (O && glitchT > 0) { ox += (R() - .5) * 10 * p.depth; oy += (R() - .5) * 4; }
      ctx.save(); ctx.translate(p.x + ox, p.y + oy); ctx.rotate(p.rot);
      ctx.beginPath(); p.pts.forEach(([x, y], i) => ctx[i ? "lineTo" : "moveTo"](x, y)); ctx.closePath();
      if (p.kind === 0) { ctx.fillStyle = D ? `rgba(40,40,40,${.35 + p.depth * .4})` : `rgba(10,10,10,${.25 + p.depth * .55})`; ctx.fill(); }
      else if (p.kind === 1) { ctx.strokeStyle = D ? "rgba(255,255,255,.22)" : "rgba(255,255,255,.55)"; ctx.lineWidth = 2 + p.depth * 2; ctx.stroke(); }
      else { ctx.fillStyle = D ? `rgba(229,0,26,${.25 + p.depth * .45})` : `rgba(120,0,14,${.35 + p.depth * .45})`; ctx.fill(); ctx.strokeStyle = "rgba(0,0,0,.5)"; ctx.lineWidth = 2; ctx.stroke(); }
      ctx.restore();
    }
    // 星星
    for (const st of stars) {
      if (F && !O) { st.x += st.vx * s; if (st.x < -20) { st.x = W + 20; st.y = R() * H; } }
      const k = .5 + .5 * Math.sin(st.ph + t * .001 * st.sp), r = st.s * (.35 + k * .75);
      starPath(st.x + mx * -14, st.y + my * -10, r, st.ph + t * .0004);
      ctx.fillStyle = D ? (st.s > 8 ? `rgba(229,0,26,${.4 + k * .5})` : `rgba(255,255,255,${.25 + k * .5})`) : `rgba(255,255,255,${.3 + k * .6})`;
      ctx.fill();
    }
    // 偶尔一道斜光
    if (F && !O) {
      nextStreak -= dt;
      if (!streak && nextStreak <= 0) { streak = { p: 0, y: H * (.2 + R() * .6) }; nextStreak = 3500 + R() * 3000; }
      if (streak) {
        streak.p += dt / 520;
        const e = 1 - Math.pow(1 - Math.min(1, streak.p), 3), len = W * .55, x = -len + (W + len * 2) * e;
        ctx.save(); ctx.translate(x, streak.y); ctx.rotate(-.33);
        const lg = ctx.createLinearGradient(-len / 2, 0, len / 2, 0), c = D ? "229,0,26" : "255,255,255";
        lg.addColorStop(0, `rgba(${c},0)`); lg.addColorStop(.7, `rgba(${c},.75)`); lg.addColorStop(1, `rgba(${c},0)`);
        ctx.fillStyle = lg; ctx.fillRect(-len / 2, -2, len, 4); ctx.fillRect(-len / 3, 9, len * .5, 2); ctx.restore();
        if (streak.p >= 1) streak = null;
      }
    }
    // 断线：横向错位条 + 扫描线
    if (O || glitchT > 0) {
      glitchT = Math.max(0, glitchT - dt);
      const n = glitchT > 0 ? 8 : 2;
      for (let i = 0; i < n; i++) {
        if (R() > (glitchT > 0 ? .2 : .85)) continue;
        const y = R() * H, h = 2 + R() * (glitchT > 0 ? 40 : 10);
        try { ctx.drawImage(cv, 0, y * dpr, W * dpr, h * dpr, (R() - .5) * 60, y, W, h); } catch (_) {}
      }
      ctx.fillStyle = "rgba(0,0,0,.18)";
      for (let y = 0; y < H; y += 4) ctx.fillRect(0, y, W, 1);
    }
    // 暗角
    const v = ctx.createRadialGradient(W / 2, H * .45, Math.min(W, H) * .35, W / 2, H / 2, Math.hypot(W, H) * .62);
    v.addColorStop(0, "rgba(0,0,0,0)"); v.addColorStop(1, D ? "rgba(0,0,0,.6)" : "rgba(40,0,4,.5)");
    ctx.fillStyle = v; ctx.fillRect(0, 0, W, H);
  }
  function loop(now) {
    raf = 0;
    const live = (full() && !off()) || glitchT > 0;
    if (!live || document.hidden) { last = 0; return; }
    const dt = Math.min(80, now - (last || now)); last = now; acc += dt;
    mx += (tmx - mx) * Math.min(1, dt / 220); my += (tmy - my) * Math.min(1, dt / 220);
    if (acc >= 32) { t += acc; draw(acc); acc = 0; }
    raf = requestAnimationFrame(loop);
  }
  const kick = () => { if (!raf) { last = 0; raf = requestAnimationFrame(loop); } };
  let rt = 0;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { init(); draw(0); }, 120); });
  addEventListener("pointermove", (e) => { tmx = e.clientX / (W || 1) - .5; tmy = e.clientY / (H || 1) - .5; }, { passive: true });
  document.addEventListener("visibilitychange", kick);
  new MutationObserver(() => { draw(0); kick(); }).observe(root, { attributes: true, attributeFilter: ["data-motion", "data-ui-theme", "class"] });
  window.P5BG = { glitch() { glitchT = 320; kick(); }, redraw() { draw(0); } };
  init(); draw(0); kick();
})();
