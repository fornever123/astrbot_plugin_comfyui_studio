/* ==========================================================
   p5-fx.js · P5 风格动效引擎（纯 Web Animations API，无依赖）
   规则：不播片 —— 没有全屏过场；所有反馈都很短（≤0.6s）且不拦截操作。
   「简化动效」（html[data-motion="lite"]）时绝大多数函数直接返回。
   ========================================================== */
(() => {
  "use strict";
  const root = document.documentElement;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const full = () => root.dataset.motion !== "lite";
  const E = { out: "cubic-bezier(.16,1,.3,1)", back: "cubic-bezier(.3,1.7,.45,1)", snap: "cubic-bezier(.75,0,.2,1)", in: "cubic-bezier(.6,0,.9,.35)" };
  const R = Math.random;
  const can = typeof Element !== "undefined" && typeof Element.prototype.animate === "function";
  const NOOP = { finished: Promise.resolve(), cancel() {} };
  const wait = (ms) => new Promise((ok) => setTimeout(ok, Math.max(0, ms)));
  const anim = (el, kf, o = {}) => (el && can ? el.animate(kf, { fill: "backwards", easing: E.out, duration: 300, ...o }) : NOOP);
  const layer = () => document.getElementById("p5Fx") || document.body;
  const mk = (cls, html = "", parent = layer()) => { const e = document.createElement("div"); e.className = cls; if (html) e.innerHTML = html; parent.appendChild(e); return e; };
  const gone = (a, el) => a.finished.then(() => el.remove(), () => el.remove());
  const center = (el) => { const r = el.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2, r }; };
  const own = (a) => !(window.CSSAnimation && a instanceof CSSAnimation) && !(window.CSSTransition && a instanceof CSSTransition);
  const kill = (el) => { if (el && el.getAnimations) el.getAnimations().filter(own).forEach((a) => a.cancel()); };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ---------- 勒索信拼贴字：textContent 不变，只把每个字包成 span ---------- */
  const STY = ["r0", "r1", "r2", "r3", "r4"];
  const hash = (s) => { let h = 2166136261; for (const c of s) { h ^= c.codePointAt(0); h = Math.imul(h, 16777619); } return h >>> 0; };
  function ransom(el, text, o = {}) {
    const t = text ?? el.textContent;
    let seed = hash(t + (o.salt || "")) || 1;
    const rr = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
    const pool = o.styles || STY;
    let last = -1;
    el.textContent = "";
    for (const c of t) {
      const s = document.createElement("span");
      if (c === " ") { s.className = "rl sp"; s.textContent = " "; el.appendChild(s); continue; }
      let k; do { k = Math.floor(rr() * pool.length); } while (k === last && pool.length > 1); last = k;
      const r = ((rr() * 2 - 1) * (o.rot ?? 8)).toFixed(1), sc = (1 + (rr() * 2 - 1) * (o.scale ?? .12)).toFixed(2), dy = ((rr() * 2 - 1) * (o.dy ?? .06)).toFixed(3);
      s.className = `rl ${pool[k]}${/[\x21-\x7e]/.test(c) ? " lat" : ""}`;
      s.style.cssText = `--r:${r}deg;--s:${sc};--dy:${dy}em`;
      Object.assign(s.dataset, { r, s: sc, dy });
      s.textContent = c;
      el.appendChild(s);
    }
    return $$(".rl:not(.sp)", el);
  }
  function ransomIn(letters, delay = 0, step = 45) {
    if (!full()) return;
    letters.forEach((s, i) => {
      const { r, s: sc, dy } = s.dataset, fin = `translateY(${dy}em) rotate(${r}deg) scale(${sc})`, side = i % 2 ? 1 : -1;
      anim(s, [
        { opacity: 0, transform: `translate(${side * (20 + R() * 30)}px, ${-30 - R() * 30}px) rotate(${+r + side * 40}deg) scale(2.4)` },
        { opacity: 1, transform: `translateY(${dy}em) rotate(${r}deg) scale(${sc * .88})`, offset: .62 },
        { opacity: 1, transform: fin }
      ], { duration: 260, delay: delay + i * step });
    });
  }

  /* ---------- 粒子：三角碎片 + 星星 ---------- */
  function burst(x, y, o = {}) {
    if (!full() || !can) return;
    const n = o.n ?? 8, d = o.dist ?? 48, dur = o.dur ?? 360, cols = o.cols || ["", "k", "w", ""];
    for (let i = 0; i < n; i++) {
      const s = 5 + R() * (o.size ?? 10), a = (i / n) * Math.PI * 2 + R() * .7, dist = d * (.55 + R() * .6);
      const el = mk(`p5-sh ${cols[i % cols.length]}${R() < .35 ? " st" : ""}`);
      el.style.width = el.style.height = s + "px";
      gone(anim(el, [
        { transform: `translate(${x - s / 2}px, ${y - s / 2}px) rotate(0deg) scale(1)`, opacity: 1 },
        { transform: `translate(${x - s / 2 + Math.cos(a) * dist}px, ${y - s / 2 + Math.sin(a) * dist}px) rotate(${(R() - .5) * 540}deg) scale(.15)`, opacity: .2 }
      ], { duration: dur * (.8 + R() * .4), fill: "forwards" }), el);
    }
    if (o.ring !== false) {
      const g = mk(o.ring2 ? "p5-ring2" : "p5-ring");
      g.style.left = x + "px"; g.style.top = y + "px";
      gone(anim(g, [{ transform: "scale(.5) rotate(0deg)", opacity: 1 }, { transform: `scale(${o.ringScale ?? 5}) rotate(30deg)`, opacity: 0 }], { duration: dur * .8, fill: "forwards" }), g);
    }
  }
  const tap = (x, y) => burst(x, y, { n: 5, dist: 26, dur: 260, size: 6, ringScale: 3.2 });

  /* ---------- 页面标题区入场 ---------- */
  function heroIn(head) {
    if (!head || !can) return;
    [head, ...$$("*", head)].forEach(kill);
    if (!full()) { anim(head, [{ opacity: 0 }, { opacity: 1 }], { duration: 140 }); return; }
    anim(head, [{ clipPath: "polygon(0 4%, 0 0, 0 100%, 0 95%)" }, { clipPath: "polygon(0 4%, 100% 0, 93% 100%, 0 95%)" }], { duration: 300, easing: E.snap, pseudoElement: "::before" });
    anim(head, [{ opacity: 0, transform: "translateX(-60px)" }, { opacity: 1, transform: "none" }], { duration: 320, delay: 50, pseudoElement: "::after" });
    anim($(".p5-wm", head), [{ opacity: 0, transform: "translateX(90px) skewX(-20deg)" }, { opacity: 1, transform: "none" }], { duration: 420, delay: 80 });
    anim($(".p5-tg", head), [{ opacity: 0, transform: "translateX(-40px) rotate(-3deg)" }, { opacity: 1, transform: "rotate(-3deg)" }], { duration: 260, delay: 80, easing: E.back });
    ransomIn($$("h2 .rl:not(.sp)", head), 110, Math.max(22, 45 - $$("h2 .rl", head).length * 2));
    anim($(".p5-en", head), [{ opacity: 0, clipPath: "inset(0 100% 0 0)", transform: "translateX(-20px)" }, { opacity: 1, clipPath: "inset(0 0 -20% 0)", transform: "none" }], { duration: 300, delay: 230, easing: E.snap });
    anim($("p", head), [{ opacity: 0, transform: "translateY(10px)" }, { opacity: 1, transform: "none" }], { duration: 260, delay: 270 });
    anim($(".p5-emb b", head), [{ opacity: 0, transform: "rotate(-40deg) scale(2.2)" }, { opacity: 1, transform: "rotate(-8deg)" }], { duration: 340, delay: 120, easing: E.back });
    anim($(".p5-emb .rays", head), [{ opacity: 0 }, { opacity: 1 }], { duration: 500 });
    $$(".p5-emb i", head).forEach((d, i) => anim(d, [{ opacity: 0, scale: 0 }, { opacity: 1, scale: 1 }], { duration: 300, delay: 300 + i * 60, easing: E.back }));
  }

  /* ---------- 面板切入（进入视口时） ---------- */
  function panelIn(el, delay = 0) {
    if (!can) return;
    if (!full()) { anim(el, [{ opacity: 0 }, { opacity: 1 }], { duration: 140, delay }); return; }
    anim(el, [
      { opacity: 0, transform: "translateX(-70px) skewX(7deg)", clipPath: "polygon(-30px -90px, -30px -90px, -30px calc(100% + 30px), -30px calc(100% + 30px))" },
      { opacity: 1, transform: "none", clipPath: "polygon(-30px -90px, calc(100% + 30px) -90px, calc(100% + 30px) calc(100% + 30px), -30px calc(100% + 30px))" }
    ], { duration: 380, delay, easing: E.snap });
    const h2 = $(":scope > .section-head h2", el), head = $(":scope > .section-head", el);
    if (h2) anim(h2, [{ opacity: 0, transform: "skewX(-10deg) rotate(-1.4deg) translateX(-40px) scaleX(1.3)" }, { opacity: 1, transform: "skewX(-10deg) rotate(-1.4deg)" }], { duration: 300, delay: delay + 140, easing: E.back });
    if (head) anim(head, [{ opacity: 0, transform: "rotate(-200deg) scale(0)" }, { opacity: 1, transform: "rotate(-9deg)" }], { duration: 360, delay: delay + 110, easing: E.back, pseudoElement: "::before" });
    $$(":scope > .form-grid > *, :scope > .folder-grid > *, :scope > .workflow-grid > *, .preset, .builtin-bundle-card, .path-setting-row", el).slice(0, 18)
      .forEach((c, i) => anim(c, [{ opacity: 0, transform: "translateY(14px)" }, { opacity: 1, transform: "none" }], { duration: 240, delay: delay + 180 + i * 16 }));
  }
  let io = null;
  function reveal(view) {
    if (io) io.disconnect();
    const bands = $$(":scope > .band, :scope > .lora-isolation-notice", view).filter((b) => !b.hidden);
    let k = 0;
    const later = [];
    bands.forEach((b) => { if (b.getBoundingClientRect().top < innerHeight - 40) panelIn(b, 140 + k++ * 70); else later.push(b); });
    if (!later.length || !("IntersectionObserver" in window)) return;
    let batch = 0, batchT = 0;
    io = new IntersectionObserver((ents) => {
      const now = performance.now(); if (now - batchT > 120) batch = 0; batchT = now;
      ents.filter((e) => e.isIntersecting).forEach((e) => { io.unobserve(e.target); panelIn(e.target, batch++ * 70); });
    }, { rootMargin: "0px 0px -6% 0px" });
    later.forEach((b) => io.observe(b));
  }

  /* ---------- 切页：一道斜切（0.3s，不挡操作） ---------- */
  function slash() {
    if (!full() || !can) return;
    const W = innerWidth, top = parseFloat(getComputedStyle(root).getPropertyValue("--top-h")) || 74, h = innerHeight - top;
    [["k", 190, 0], ["r", 70, 40], ["w", 14, 70]].forEach(([c, w, d]) => {
      const el = mk("p5-slash " + c); el.style.width = w + "px"; el.style.top = top + "px"; el.style.height = h + "px";
      gone(anim(el, [{ transform: `translateX(${-w - h * .4}px) skewX(-24deg)` }, { transform: `translateX(${W + h * .4}px) skewX(-24deg)` }], { duration: 300, delay: d, easing: E.snap, fill: "both" }), el);
    });
  }

  /* ---------- 小部件 ---------- */
  function shake(el, amp = 7, ms = 320) {
    if (!full() || !el) return;
    const k = [];
    for (let i = 0; i <= 8; i++) { const a = amp * (1 - i / 8); k.push({ translate: i === 8 ? "0 0" : `${(i % 2 ? -1 : 1) * a}px ${(i % 3 - 1) * a * .4}px` }); }
    anim(el, k, { duration: ms, easing: "linear", fill: "none" });
  }
  function squish(el) { if (full() && el) anim(el, [{ scale: "1 1" }, { scale: "1.14 .8", offset: .3 }, { scale: ".96 1.06", offset: .65 }, { scale: "1 1" }], { duration: 320, easing: "ease-out" }); }
  function swSpark(input) {
    if (!full() || !input) return;
    const r = input.getBoundingClientRect();
    if (input.checked) burst(r.right - 10, r.top + r.height / 2, { n: 7, dist: 34, dur: 300, size: 7, ringScale: 3.6 });
    else burst(r.left + 10, r.top + r.height / 2, { n: 3, dist: 18, dur: 220, size: 5, cols: ["k"], ring: false });
  }
  function dealIn(cards) {
    if (!full()) return;
    cards.slice(0, 24).forEach((el, i) => anim(el, [{ opacity: 0, transform: "translate(80px, 0) rotate(6deg) skewX(-12deg)" }, { opacity: 1, transform: "none" }], { duration: 300, delay: i * 30, easing: E.back }));
  }

  /* ---------- 成功：碎片 + 拼贴字小贴纸（不全屏） ---------- */
  function stamp(btn, msg = "已保存！") {
    if (!full() || !btn || !btn.isConnected) return;
    const c = center(btn);
    if (!c.r.width) return;
    burst(c.x, c.y, { n: 14, dist: 96, dur: 460, size: 13, ringScale: 8, ring2: true });
    anim(btn, [{ scale: 1 }, { scale: 1.12, offset: .25 }, { scale: .97, offset: .6 }, { scale: 1 }], { duration: 320 });
    const tag = mk("p5-tagpop");
    ransom(tag, msg, { rot: 10 });
    const tw = tag.offsetWidth, th = tag.offsetHeight;
    tag.style.left = Math.min(innerWidth - tw - 10, Math.max(10, c.x - tw / 2)) + "px";
    tag.style.top = Math.max(10, c.r.top - th - 14) + "px";
    anim(tag, [{ opacity: 0, transform: "translateY(26px) scale(.3) rotate(-16deg)" }, { opacity: 1, transform: "translateY(-6px) scale(1.12) rotate(-4deg)", offset: .55 }, { opacity: 1, transform: "rotate(-4deg)" }], { duration: 300, fill: "forwards" });
    ransomIn($$(".rl:not(.sp)", tag), 40, 35);
    wait(1000).then(() => gone(anim(tag, [{ opacity: 1, transform: "rotate(-4deg)" }, { opacity: 0, transform: "translate(40px,-18px) rotate(4deg) scale(.9)" }], { duration: 200, easing: E.in, fill: "forwards" }), tag));
  }
  /* 出错：按钮抖一下 + 小碎片 + 气泡 */
  function oops(btn, text) {
    if (!full() || !btn || !btn.isConnected) return;
    shake(btn, 8, 340);
    const r = btn.getBoundingClientRect();
    if (!r.width) return;
    burst(r.right - 10, r.top, { n: 6, dist: 38, dur: 300, size: 8, cols: ["", "k"] });
    if (!text) return;
    const b = mk("p5-bubble", esc(text.length > 46 ? text.slice(0, 45) + "…" : text));
    b.style.left = Math.min(innerWidth - b.offsetWidth - 10, Math.max(10, r.left)) + "px";
    b.style.top = Math.max(10, r.top - b.offsetHeight - 14) + "px";
    anim(b, [{ opacity: 0, transform: "scale(.3) rotate(-16deg) translateY(20px)" }, { opacity: 1, transform: "scale(1.1) rotate(-3deg)", offset: .6 }, { opacity: 1, transform: "rotate(-3deg)" }], { duration: 280, fill: "forwards" });
    wait(1700).then(() => gone(anim(b, [{ opacity: 1, transform: "rotate(-3deg)" }, { opacity: 0, transform: "rotate(4deg) translateY(-14px) scale(.9)" }], { duration: 180, fill: "forwards" }), b));
  }
  /* 进度完成小章 */
  function doneStamp(box) {
    if (!box) return;
    const c = center(box);
    if (!c.r.width) return;
    burst(c.r.right - 60, c.r.top + 18, { n: 10, dist: 60, dur: 400, size: 10, ringScale: 6, ring2: true });
    const tag = mk("p5-tagpop");
    ransom(tag, "完成！", { rot: 12 });
    tag.style.left = c.r.right - tag.offsetWidth - 30 + "px";
    tag.style.top = c.r.top - tag.offsetHeight / 2 + "px";
    anim(tag, [{ opacity: 0, transform: "scale(2.6) rotate(-30deg)" }, { opacity: 1, transform: "scale(.9) rotate(-6deg)", offset: .7 }, { opacity: 1, transform: "rotate(-8deg)" }], { duration: 300, easing: "ease-in", fill: "forwards" });
    wait(1200).then(() => gone(anim(tag, [{ opacity: 1 }, { opacity: 0, transform: "translateY(-14px) rotate(-8deg)" }], { duration: 200, fill: "forwards" }), tag));
  }
  /* 提示条进场 */
  function toastIn(el, kind) {
    if (!full() || !el) return;
    anim(el, [{ transform: "translateX(-18px) skewX(-8deg)" }, { transform: "skewX(-8deg)" }], { duration: 260, easing: E.back });
    if (kind === "p5-err") shake(el, 6, 300);
  }

  /* ---------- 断线：顶栏抖一下 + 背景转灰（状态，不是过场） ---------- */
  function offline(on) {
    root.classList.toggle("p5-off", on);
    if (window.P5BG) window.P5BG.glitch();
    if (!on || !full() || !can) return;
    anim($(".sidebar"), [{ transform: "none", filter: "none" }, { transform: "translateX(-8px) skewX(5deg)", filter: "hue-rotate(80deg) saturate(3)", offset: .2 }, { transform: "translateX(6px)", filter: "invert(1)", offset: .4 }, { transform: "translateX(-3px)", filter: "none", offset: .65 }, { transform: "none", filter: "none" }], { duration: 300, easing: "steps(6)" });
    [["k", 18, .22], ["r", 8, .5], ["w", 4, .7]].forEach(([c, h, y], i) => {
      const el = mk("p5-slash " + c); el.style.cssText = `left:0;width:100%;top:${innerHeight * y}px;height:${h}px`;
      gone(anim(el, [{ transform: "translateX(-100%)", opacity: 1 }, { transform: "translateX(0)", opacity: 1, offset: .5 }, { transform: "translateX(100%)", opacity: 0 }], { duration: 260, delay: i * 30, easing: "steps(4)", fill: "both" }), el);
    });
  }

  /* ---------- 主题切换：锯齿斜切擦除 ---------- */
  function themeSwap(apply) {
    if (!document.startViewTransition || !full()) { apply(); return; }
    let vt;
    try { vt = document.startViewTransition(apply); } catch (_) { apply(); return; }
    const p = (x1, x2) => `polygon(${x1}% 0%, 160% 0%, 160% 100%, ${x2}% 100%, ${(x1 + x2) / 2 - 7}% 50%)`;
    vt.ready.then(() => root.animate({ clipPath: [p(110, 140), p(-70, -40)] }, { duration: 520, easing: E.snap, pseudoElement: "::view-transition-new(root)" })).catch(() => {});
  }

  /* ---------- Ctrl K 跳转后：框出目标 ---------- */
  function pointAt(el) {
    if (!el) return;
    el.scrollIntoView({ block: "center", behavior: "instant" });
    if (!full() || !can) { el.classList.add("p5-hl"); wait(1400).then(() => el.classList.remove("p5-hl")); return; }
    const r = el.getBoundingClientRect(), fr = mk("p5-ptframe"), pad = 8;
    fr.style.cssText = `left:${r.left - pad}px;top:${r.top - pad}px;width:${r.width + pad * 2}px;height:${r.height + pad * 2}px`;
    anim(fr, [{ opacity: 0, transform: "scale(1.5) rotate(-4deg)" }, { opacity: 1, transform: "none" }], { duration: 260, easing: E.back, fill: "forwards" });
    const tg = mk("p5-pttag", "在这里！");
    tg.style.left = Math.max(6, r.left - 14) + "px"; tg.style.top = Math.max(6, r.top - pad - 34) + "px";
    anim(tg, [{ opacity: 0, transform: "rotate(-30deg) scale(2.2)" }, { opacity: 1, transform: "rotate(-6deg)" }], { duration: 260, delay: 80, easing: E.back, fill: "both" });
    wait(160).then(() => { const c = center(tg); burst(c.x, c.y, { n: 8, dist: 46, dur: 340, size: 9 }); });
    wait(1700).then(() => [fr, tg].forEach((e) => gone(anim(e, [{ opacity: 1 }, { opacity: 0 }], { duration: 200, fill: "forwards" }), e)));
  }

  window.P5FX = { full, wait, anim, E, kill, ransom, ransomIn, burst, tap, heroIn, panelIn, reveal, slash, shake, squish, swSpark, dealIn, stamp, oops, doneStamp, toastIn, offline, themeSwap, pointAt };
})();
