/* ==========================================================
   p5-ui.js · 把 P5 皮肤接到现有控制台上（不改 app.js 的任何逻辑）
   - 标题拼贴字、页码、英文副标题（只改展示，textContent 不变）
   - 切页斜切 + 标题区/面板入场；LoRA 分区切换发牌
   - 主题切换走 View Transition 锯齿擦除
   - 包一层 window.show：按内容给提示条分 成功/提醒/出错，并给刚点的按钮盖章或抖动
   - 观察 #status：未连接时背景转灰 + 断线提示条
   - 字段标签聚焦变黑底斜标；按钮点击白光 + 碎片；开关星星；卡片 3D 倾斜
   - Ctrl K 快速跳转到任意页面、面板或设置项
   - 「满配 / 简化」动效开关（sandbox iframe 里 localStorage 可能不可用，一律 try/catch）
   ========================================================== */
(() => {
  "use strict";
  const FX = window.P5FX;
  if (!FX) return;
  const root = document.documentElement;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const MOTION_KEY = "comfyui-ai-studio-motion";
  const VIEW_EN = { txt2img: "TEXT TO IMAGE", img2img: "IMAGE TO IMAGE", lora: "LORA GROUPS", presets: "PROMPT PRESETS", artist: "ARTIST TAGS", workflow: "WORKFLOWS", paths: "FOLDERS", ai: "AI & REPLIES" };
  const VIEW_WM = { txt2img: "DRAW", img2img: "EDIT", lora: "LORA", presets: "TAGS", artist: "STYLE", workflow: "FLOW", paths: "PATHS", ai: "AI" };
  const store = {
    get(k) { try { return localStorage.getItem(k) || ""; } catch (_) { return ""; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (_) {} }
  };
  const text = (el) => (el ? el.textContent.replace(/\s+/g, " ").trim() : "");

  /* ---------- 动效档位 ---------- */
  const motionBtn = $("#p5Motion");
  function setMotion(mode) {
    root.dataset.motion = mode === "lite" ? "lite" : "full";
    if (motionBtn) {
      const lite = root.dataset.motion === "lite";
      const label = $(".p5-ib-t", motionBtn);
      if (label) label.textContent = lite ? "简化" : "满配";
      motionBtn.setAttribute("aria-pressed", lite ? "true" : "false");
      motionBtn.title = lite ? "当前：简化动效（背景静止、只留必要反馈）。点击切换到满配" : "当前：满配动效。点击切换到简化（更省电）";
    }
  }
  const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  setMotion(store.get(MOTION_KEY) || (reduce ? "lite" : "full"));
  if (motionBtn) motionBtn.addEventListener("click", () => {
    const next = root.dataset.motion === "lite" ? "full" : "lite";
    setMotion(next); store.set(MOTION_KEY, next);
    if (next === "full") { const r = motionBtn.getBoundingClientRect(); FX.burst(r.left + r.width / 2, r.top + r.height / 2, { n: 10, dist: 50, dur: 380, size: 9 }); }
  });

  /* ---------- 导航分组：横排后始终展开 ---------- */
  $$(".nav-group").forEach((d) => {
    d.open = true;
    const sum = $("summary", d);
    if (sum) { sum.setAttribute("aria-expanded", "true"); sum.tabIndex = -1; sum.addEventListener("click", (e) => e.preventDefault()); }
  });

  /* ---------- 品牌拼贴字 ---------- */
  const brand = $(".brand h1");
  if (brand && !brand.dataset.p5) { brand.dataset.p5 = "1"; brand.setAttribute("aria-label", text(brand)); FX.ransom(brand, text(brand), { rot: 7, styles: ["r0", "r1", "r2", "r0", "r4"] }); }

  /* ---------- 页面标题区：页码 + 分组 + 拼贴字 + 英文副标题 ---------- */
  const order = $$("[data-view-tab]").map((t) => t.dataset.viewTab);
  $$("[data-view-panel]").forEach((panel) => {
    const head = $(":scope > .page-head", panel), h2 = head && $("h2", head);
    if (!head || !h2 || head.dataset.p5) return;
    head.dataset.p5 = "1";
    const name = panel.dataset.viewPanel, no = String(Math.max(0, order.indexOf(name)) + 1).padStart(2, "0");
    const tab = $(`[data-view-tab="${name}"]`), group = text(tab && tab.closest(".nav-group") && $(".nav-group-head span", tab.closest(".nav-group")));
    const tg = document.createElement("div");
    tg.className = "p5-tg"; tg.setAttribute("aria-hidden", "true");
    tg.innerHTML = `<b>PAGE ${no}</b>${group ? "<em></em>" : ""}`;
    if (group) $("em", tg).textContent = group;
    head.insertBefore(tg, h2);
    const title = text(h2);
    h2.setAttribute("aria-label", title);
    FX.ransom(h2, title, { salt: name });
    const en = document.createElement("div");
    en.className = "p5-en"; en.setAttribute("aria-hidden", "true"); en.textContent = VIEW_EN[name] || name.toUpperCase();
    h2.after(en);
    const wm = document.createElement("div");
    wm.className = "p5-wm"; wm.setAttribute("aria-hidden", "true"); wm.textContent = VIEW_WM[name] || "";
    const emb = document.createElement("div");
    emb.className = "p5-emb"; emb.setAttribute("aria-hidden", "true");
    emb.innerHTML = `<span class="rays"></span><b>${no}</b><i class="d1"></i><i class="d2"></i><i class="d3"></i>`;
    head.append(wm, emb);
  });

  /* ---------- 字段标签：前导文字包成 .lb（聚焦时变黑底斜标） ---------- */
  const SKIP_LABEL = ".check, .upload-button, .engine-lora-row, .lora-check, .lora-simple-switch, .style-simple-weight";
  function wrapLabels(scope) {
    const list = scope.matches && scope.matches("label") ? [scope] : $$("label", scope);
    list.forEach((lab) => {
      if (lab.matches(SKIP_LABEL) || !lab.closest(".main")) return;
      const first = lab.firstChild;
      if (!first || first.nodeType !== 3 || !first.textContent.trim()) return;
      const s = document.createElement("span");
      s.className = "lb"; s.textContent = first.textContent.trim();
      lab.replaceChild(s, first);
    });
  }
  wrapLabels(document);
  let pending = new Set(), wrapRaf = 0;
  const main = $(".main");
  if (main && "MutationObserver" in window) {
    new MutationObserver((muts) => {
      for (const m of muts) for (const n of m.addedNodes) if (n.nodeType === 1) pending.add(n);
      if (pending.size && !wrapRaf) wrapRaf = requestAnimationFrame(() => { wrapRaf = 0; const items = [...pending]; pending = new Set(); items.forEach((n) => n.isConnected && wrapLabels(n)); });
    }).observe(main, { childList: true, subtree: true });
  }

  /* ---------- 切页：斜切 + 标题区 + 面板依次切入 ---------- */
  let shown = null;
  function onView(panel, first) {
    if (!panel || panel === shown) return;
    shown = panel;
    if (!first) FX.slash();
    FX.heroIn($(":scope > .page-head", panel));
    FX.reveal(panel);
  }
  const panels = $$("[data-view-panel]");
  if ("MutationObserver" in window) {
    const mo = new MutationObserver(() => { const p = panels.find((x) => !x.hidden); if (p) onView(p, false); });
    panels.forEach((p) => mo.observe(p, { attributes: true, attributeFilter: ["hidden"] }));
  }
  const ready = document.fonts && document.fonts.ready ? Promise.race([document.fonts.ready, FX.wait(600)]) : FX.wait(0);
  ready.then(() => onView(panels.find((x) => !x.hidden), true));

  /* ---------- LoRA 分区：切换后发牌 ---------- */
  document.addEventListener("click", (e) => {
    const tab = e.target.closest && e.target.closest("[data-lora-section]");
    if (!tab) return;
    requestAnimationFrame(() => {
      const band = $$("#view-lora > .band").find((b) => !b.hidden && b.matches(".lora-management-band, .style-lora-band, .engine-lora-band"));
      if (band) FX.dealIn($$(".lora-card, .engine-lora-row", band));
    });
  });

  /* ---------- 主题切换：锯齿擦除（保留 app.js 的 setTheme 逻辑） ---------- */
  const themeBtn = $("#themeToggle");
  if (themeBtn && typeof themeBtn.onclick === "function") {
    const apply = themeBtn.onclick;
    themeBtn.onclick = function (ev) { FX.themeSwap(() => apply.call(themeBtn, ev)); };
  }

  /* ---------- 记住刚点过的按钮（给提示条找“主人”） ---------- */
  let lastBtn = null, lastAt = 0;
  const TAP = "button, .upload-button, .page-tab, .lora-section-tab";
  document.addEventListener("pointerdown", (e) => {
    const b = e.target.closest && e.target.closest(TAP);
    if (!b) return;
    lastBtn = b; lastAt = performance.now();
    if (b.matches("button.primary, button.secondary, .preset .button-row > button")) {
      b.classList.remove("hit"); void b.offsetWidth; b.classList.add("hit");
      setTimeout(() => b.classList.remove("hit"), 340);
    }
    FX.tap(e.clientX, e.clientY);
  }, true);
  document.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && e.target.matches && e.target.matches("button")) { lastBtn = e.target; lastAt = performance.now(); }
  }, true);

  /* ---------- 提示条：分类 + 盖章 / 抖动 ---------- */
  const toast = $("#toast");
  const ERR = /失败|错误|异常|无法|不能|不支持|超时|拒绝|找不到|未找到|不存在|无效|非法|出错|error|failed|denied|timeout|HTTP\s?\d{3}/i;
  const WARN = /^请|请先|请输入|请填写|请选择|没有需要|尚未|已取消|未选择|为空/;
  const kindOf = (msg) => (ERR.test(msg) ? "p5-err" : WARN.test(msg) ? "p5-warn" : "p5-ok");
  if (toast && typeof window.show === "function") {
    const orig = window.show;
    window.show = function (message) {
      const msg = String(message ?? "");
      const kind = kindOf(msg), was = toast.classList.contains("show");
      toast.classList.remove("p5-ok", "p5-err", "p5-warn");
      toast.classList.add(kind);
      const out = orig.apply(this, arguments);
      if (was) FX.toastIn(toast, kind);
      const btn = performance.now() - lastAt < 8000 ? lastBtn : null;
      if (btn && btn.isConnected) {
        if (kind === "p5-err") FX.oops(btn, msg);
        else if (kind === "p5-warn") FX.shake(btn, 5, 260);
        else if (/保存|添加|启用|应用|导入|下载|上传|纳入|恢复|删除|移除/.test(text(btn))) FX.stamp(btn, /删除|移除/.test(text(btn)) ? "搞定！" : "已保存！");
        lastBtn = null;
      }
      return out;
    };
  }

  /* ---------- ComfyUI 连接状态：断线转灰 ---------- */
  const status = $("#status"), offText = $("#p5OffText");
  if (status && "MutationObserver" in window) {
    let was = "";
    const sync = () => {
      const bad = status.classList.contains("bad"), ok = status.classList.contains("ok"), now = bad ? "bad" : ok ? "ok" : "";
      if (now === was) return;
      was = now;
      if (offText) offText.textContent = text($("#statusText")) || "ComfyUI 未连接";
      FX.offline(bad);
      if (ok) { const r = status.getBoundingClientRect(); FX.burst(r.left + 14, r.top + r.height / 2, { n: 6, dist: 30, dur: 280, size: 7, ringScale: 3 }); }
    };
    new MutationObserver(sync).observe(status, { attributes: true, attributeFilter: ["class"] });
    sync();
  }

  /* ---------- 开关：星星 ---------- */
  document.addEventListener("change", (e) => {
    const i = e.target;
    if (i && i.matches && i.matches('input[type="checkbox"]') && i.closest("label")) FX.swSpark(i);
  });

  /* ---------- 进度条到 100%：盖“完成！” ---------- */
  $$(".download-progress").forEach((box) => {
    const pct = $(".download-progress-head span", box);
    if (!pct || !("MutationObserver" in window)) return;
    let done = false;
    new MutationObserver(() => {
      const full = /^100\s*%$/.test(text(pct));
      if (full && !done && !box.hidden) FX.doneStamp(box);
      done = full;
    }).observe(pct, { childList: true, characterData: true, subtree: true });
  });

  /* ---------- LoRA 简洁卡：3D 倾斜 + 反光（事件委托，卡片重绘也不用重新绑定） ---------- */
  let tiltCard = null, tx = 0, ty = 0, cx = 0, cy = 0, tiltRaf = 0, tiltOn = false, tiltLast = 0;
  function tiltStep(now) {
    tiltRaf = 0;
    if (!tiltCard) return;
    const dt = Math.min(64, now - (tiltLast || now)); tiltLast = now;
    const k = 1 - Math.exp(-dt / 90);
    cx += (tx - cx) * k; cy += (ty - cy) * k;
    if (tiltOn || Math.abs(cx) + Math.abs(cy) > .002) {
      tiltCard.style.transform = `perspective(900px) rotateX(${(-cy * 10).toFixed(2)}deg) rotateY(${(cx * 12).toFixed(2)}deg)`;
      tiltCard.style.setProperty("--px", `${(-cx * 10).toFixed(1)}px`);
      tiltCard.style.setProperty("--py", `${(-cy * 10).toFixed(1)}px`);
      tiltRaf = requestAnimationFrame(tiltStep);
    } else {
      tiltCard.style.transform = ""; tiltCard.style.removeProperty("--px"); tiltCard.style.removeProperty("--py");
      tiltCard.classList.remove("p5-tilt"); tiltCard = null; tiltLast = 0;
    }
  }
  const tiltKick = () => { if (!tiltRaf) tiltRaf = requestAnimationFrame(tiltStep); };
  document.addEventListener("pointermove", (e) => {
    if (!FX.full() || e.pointerType === "touch") return;
    const card = e.target.closest && e.target.closest(".lora-card-simple");
    if (tiltCard && card !== tiltCard) { tiltCard.classList.remove("p5-tilting"); tx = ty = 0; tiltOn = false; tiltKick(); if (!card) return; tiltCard.style.transform = ""; tiltCard.classList.remove("p5-tilt"); cx = cy = 0; }
    if (!card) return;
    if (e.target.closest("input, select, textarea, button")) { tx = ty = 0; tiltOn = false; card.classList.remove("p5-tilting"); tiltKick(); return; }
    const r = card.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    tiltCard = card; card.classList.add("p5-tilt", "p5-tilting");
    tx = x - .5; ty = y - .5; tiltOn = true;
    card.style.setProperty("--mx", (1 - x) * 100 + "%");
    tiltKick();
  }, { passive: true });

  /* ---------- Ctrl K 快速跳转 ---------- */
  const pal = document.createElement("div");
  pal.className = "p5-pal"; pal.setAttribute("aria-hidden", "true");
  pal.innerHTML = '<div class="p5-pal-bg"></div><div class="p5-pal-box" role="dialog" aria-modal="true" aria-label="快速跳转"><div class="p5-pal-h"><span class="p5-pal-tag">跳转</span><input type="text" placeholder="输入页面、面板或设置项，例如：步数、审核、LoRA" aria-label="搜索设置项" autocomplete="off" spellcheck="false"><kbd>Esc</kbd></div><div class="p5-pal-list" role="listbox"><i class="p5-pal-hl"></i></div><div class="p5-pal-f"><span>↑ ↓ 选择</span><span>回车 跳转</span><span>Ctrl K 打开 / 关闭</span></div></div>';
  document.body.appendChild(pal);
  const palIn = $("input", pal), palList = $(".p5-pal-list", pal), palHl = $(".p5-pal-hl", pal);
  let items = [], hits = [], cur = 0;
  const LORA_BANDS = ".lora-management-band, .style-lora-band, [data-engine-lora-panel]";
  function hiddenByOther(el) {
    for (let p = el; p && p !== document.body; p = p.parentElement) {
      if (p.hidden && !p.matches("[data-view-panel]") && !p.matches(LORA_BANDS)) return true;
    }
    return false;
  }
  function buildIndex() {
    const list = [];
    $$("[data-view-tab]").forEach((tab) => list.push({ kind: "PAGE", title: text(tab), sub: "页面", view: tab.dataset.viewTab }));
    $$("[data-view-panel]").forEach((panel) => {
      const view = panel.dataset.viewPanel, page = text($(`[data-view-tab="${view}"]`)) || view;
      $$("[data-lora-section]", panel).forEach((t) => list.push({ kind: "TAB", title: text(t), sub: `${page} · 分区`, view, section: t, el: t }));
      $$(".band", panel).forEach((band) => {
        const h = $(".section-head h2", band), bandTitle = text(h);
        if (h) list.push({ kind: "PANEL", title: bandTitle, sub: page, view, el: band, band });
        $$("label", band).forEach((lab) => {
          if (lab.closest(".lora-card, .engine-lora-list, .path-settings") || hiddenByOther(lab)) return;
          const lb = $(":scope > .lb", lab);
          let t = lb ? text(lb) : [...lab.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join(" ").replace(/\s+/g, " ").trim();
          if (!t) t = text(lab).slice(0, 40);
          if (!t) return;
          list.push({ kind: "SET", title: t.length > 46 ? t.slice(0, 45) + "…" : t, sub: `${page} · ${bandTitle}`, view, el: lab, band, focus: $("input, select, textarea", lab) });
        });
        $$("button[id]", band).forEach((b) => {
          if (b.closest(".lora-card") || hiddenByOther(b)) return;
          list.push({ kind: "BTN", title: text(b), sub: `${page} · ${bandTitle}`, view, el: b, band });
        });
      });
    });
    return list;
  }
  function score(it, q) {
    if (!q) return it.kind === "PAGE" ? 3 : 0;
    const t = it.title.toLowerCase(), s = it.sub.toLowerCase();
    if (t === q) return 10;
    if (t.startsWith(q)) return 8;
    if (t.includes(q)) return 6;
    if ((t + " " + s).includes(q)) return 3;
    const parts = q.split(/\s+/).filter(Boolean);
    return parts.length > 1 && parts.every((p) => (t + " " + s).includes(p)) ? 2 : 0;
  }
  function renderPal() {
    const q = palIn.value.trim().toLowerCase();
    hits = items.map((it, i) => ({ it, s: score(it, q), i })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s || a.i - b.i).slice(0, 40).map((x) => x.it);
    cur = Math.min(cur, Math.max(0, hits.length - 1));
    $$(".p5-pi, .p5-pal-empty", palList).forEach((n) => n.remove());
    if (!hits.length) { const e = document.createElement("div"); e.className = "p5-pal-empty"; e.textContent = "没有找到匹配的设置项"; palList.appendChild(e); palHl.style.opacity = "0"; return; }
    palHl.style.opacity = "";
    hits.forEach((it, i) => {
      const row = document.createElement("div");
      row.className = "p5-pi"; row.setAttribute("role", "option"); row.dataset.i = i;
      row.innerHTML = "<i></i><div><b></b><small></small></div><kbd>回车</kbd>";
      $("i", row).textContent = it.kind; $("b", row).textContent = it.title; $("small", row).textContent = it.sub;
      palList.appendChild(row);
    });
    setCur(cur, true);
  }
  function setCur(i, instant) {
    const rows = $$(".p5-pi", palList);
    if (!rows.length) return;
    cur = (i + rows.length) % rows.length;
    rows.forEach((r, k) => { r.classList.toggle("act", k === cur); r.setAttribute("aria-selected", k === cur ? "true" : "false"); });
    if (instant) palHl.style.transition = "none";
    palHl.style.top = rows[cur].offsetTop + "px";
    if (instant) { void palHl.offsetWidth; palHl.style.transition = ""; }
    const r = rows[cur], box = palList;
    if (r.offsetTop < box.scrollTop) box.scrollTop = r.offsetTop - 6;
    else if (r.offsetTop + r.offsetHeight > box.scrollTop + box.clientHeight) box.scrollTop = r.offsetTop + r.offsetHeight - box.clientHeight + 6;
  }
  function openPal() {
    items = buildIndex(); palIn.value = ""; cur = 0;
    pal.classList.add("on"); pal.setAttribute("aria-hidden", "false");
    renderPal(); palIn.focus();
    FX.anim($(".p5-pal-box", pal), [{ opacity: 0, transform: "translateX(-50%) rotate(-6deg) scale(.86) translateY(-20px)" }, { opacity: 1, transform: "translateX(-50%) rotate(-1deg)" }], { duration: FX.full() ? 240 : 1, easing: FX.E.back });
  }
  function closePal() { pal.classList.remove("on"); pal.setAttribute("aria-hidden", "true"); }
  function go(it) {
    closePal();
    if (!it) return;
    const tab = $(`[data-view-tab="${it.view}"]`);
    if (tab && !tab.classList.contains("active")) tab.click();
    const band = it.band;
    if (band && band.hidden && band.matches(LORA_BANDS)) {
      const sec = band.matches(".style-lora-band") ? "style" : band.matches(".lora-management-band") ? "normal" : band.dataset.engineLoraPanel;
      const t = $(`[data-lora-section="${sec}"]`); if (t) t.click();
    }
    if (it.section) it.section.click();
    if (!it.el) { window.scrollTo({ top: 0, behavior: "instant" }); return; }
    setTimeout(() => {
      FX.pointAt(it.el);
      if (it.focus && !it.focus.disabled) { try { it.focus.focus({ preventScroll: true }); } catch (_) {} }
    }, FX.full() ? 360 : 30);
  }
  palIn.addEventListener("input", () => { cur = 0; renderPal(); });
  palIn.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); setCur(cur + (e.key === "ArrowDown" ? 1 : -1)); }
    else if (e.key === "Enter") { e.preventDefault(); go(hits[cur]); }
    else if (e.key === "Escape") { e.preventDefault(); closePal(); }
  });
  palList.addEventListener("pointermove", (e) => { const r = e.target.closest(".p5-pi"); if (r && +r.dataset.i !== cur) setCur(+r.dataset.i); });
  palList.addEventListener("click", (e) => { const r = e.target.closest(".p5-pi"); if (r) go(hits[+r.dataset.i]); });
  $(".p5-pal-bg", pal).addEventListener("click", closePal);
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      if (pal.classList.contains("on")) closePal(); else openPal();
    }
  });
  const palBtn = $("#p5PaletteBtn");
  if (palBtn) palBtn.addEventListener("click", openPal);

  window.P5UI = { setMotion, openPalette: openPal, closePalette: closePal, wrapLabels };
})();
