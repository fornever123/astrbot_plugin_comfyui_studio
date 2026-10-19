/* ==========================================================
   p5-forms.js · 表单小件增强（只加展示和便捷操作，不改 app.js 的读写逻辑）
   - 数字框：右侧 − / + 按钮（按住连发），有上下限的参数再加一条滑杆
   - 多行文本框：右上角实时显示字数 / 标签数 / 名单项数
   - 文件夹卡片：一键复制路径
   - 页签小红点：该页有未保存的修改时亮起
   所有新增控件都不带 id / data-*，app.js 的取值和保存逻辑完全不受影响。
   （样式表里也用到这些字：未保存、读取中、可把图片拖到这里替换预览图，写在这里方便字体子集收录）
   ========================================================== */
(() => {
  "use strict";
  const main = document.querySelector(".main");
  if (!main) return;
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const make = (tag, cls, attrs) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (attrs) Object.keys(attrs).forEach((k) => el.setAttribute(k, attrs[k]));
    return el;
  };
  const icon = (id) => `<svg class="ui-ic" aria-hidden="true"><use href="#i-${id}"/></svg>`;
  const tell = (msg) => { try { if (typeof window.show === "function") window.show(msg); } catch (_) {} };

  /* ---------- 监听 value：app.js 回填配置时同步滑杆 / 计数 ---------- */
  const watchers = new WeakMap();
  function watchValue(el, fn) {
    let list = watchers.get(el);
    if (!list) {
      list = [];
      watchers.set(el, list);
      const run = () => list.forEach((f) => { try { f(); } catch (_) {} });
      const desc = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value");
      if (desc && desc.get && desc.set) {
        try {
          Object.defineProperty(el, "value", {
            configurable: true,
            enumerable: true,
            get() { return desc.get.call(this); },
            set(v) { desc.set.call(this, v); run(); }
          });
        } catch (_) {}
      }
      el.addEventListener("input", run);
      el.addEventListener("change", run);
    }
    list.push(fn);
  }

  /* ---------- 数字框：− / + 按钮 + 滑杆 ---------- */
  const SLIDER_IDS = new Set([
    "style_lora_random_count", "qwen21_turbo_lora_strength", "steps", "cfg", "denoise",
    "hires_scale", "hires_steps", "hires_denoise", "img2img_lora_strength", "img2img_nsfw_lora_strength",
    "img2img_steps", "img2img_cfg", "img2img_denoise", "moderation_max_side"
  ]);
  const NUM_SEL = '.form-grid > label > input[type="number"], .lora-controls > label > input[type="number"]';
  const done = new WeakSet();
  const fire = (el, type) => el.dispatchEvent(new Event(type, { bubbles: true }));
  const decimals = (x) => { const s = String(x || ""); const i = s.indexOf("."); return i < 0 ? 0 : s.length - i - 1; };

  function stepOnce(inp, dir) {
    if (inp.disabled || inp.readOnly) return false;
    const before = inp.value;
    try {
      if (dir > 0) inp.stepUp(); else inp.stepDown();
    } catch (_) {
      const step = parseFloat(inp.step) || 1, lo = parseFloat(inp.min), hi = parseFloat(inp.max);
      let v = (parseFloat(inp.value) || 0) + dir * step;
      if (!isNaN(lo)) v = Math.max(lo, v);
      if (!isNaN(hi)) v = Math.min(hi, v);
      inp.value = String(+v.toFixed(Math.max(decimals(inp.step), decimals(inp.min))));
    }
    if (inp.value === before) return false;
    fire(inp, "input");
    return true;
  }

  function bindStep(btn, inp, dir) {
    let wait = 0, rep = 0, changed = false;
    const stop = () => {
      clearTimeout(wait); clearInterval(rep); wait = rep = 0;
      btn.classList.remove("on");
      if (changed) { changed = false; fire(inp, "change"); }
    };
    btn.addEventListener("mousedown", (e) => e.preventDefault()); // 不抢输入框焦点
    btn.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || inp.disabled) return;
      e.preventDefault();
      stop();
      btn.classList.add("on");
      changed = stepOnce(inp, dir) || changed;
      try { btn.setPointerCapture(e.pointerId); } catch (_) {}
      wait = setTimeout(() => { rep = setInterval(() => { changed = stepOnce(inp, dir) || changed; }, 55); }, 380);
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach((t) => btn.addEventListener(t, stop));
    btn.addEventListener("click", (e) => e.preventDefault());
  }

  function enhanceNumber(inp) {
    if (done.has(inp) || inp.closest(".p5-num")) return;
    done.add(inp);
    const wrap = make("span", "p5-num");
    inp.parentNode.insertBefore(wrap, inp);
    wrap.appendChild(inp); // 输入框保持在标签内第一个可聚焦控件的位置，点击标签文字仍然聚焦它
    const dec = make("button", "p5-step dec", { type: "button", tabindex: "-1", "aria-label": "减少", title: "减少（按住连续调整）" });
    const inc = make("button", "p5-step inc", { type: "button", tabindex: "-1", "aria-label": "增加", title: "增加（按住连续调整）" });
    wrap.append(dec, inc);
    bindStep(dec, inp, -1);
    bindStep(inc, inp, 1);

    let range = null;
    const lo = parseFloat(inp.min), hi = parseFloat(inp.max);
    if ((SLIDER_IDS.has(inp.id) || inp.matches(".lora-controls input.weight")) && !isNaN(lo) && !isNaN(hi) && hi > lo) {
      range = make("input", "p5-range", { type: "range", tabindex: "-1", "aria-hidden": "true" });
      range.min = String(lo); range.max = String(hi); range.step = inp.step || "any";
      const ticks = make("span", "p5-range-wrap");
      ticks.appendChild(range);
      wrap.after(ticks);
      range.addEventListener("input", (e) => {
        e.stopPropagation();
        inp.value = range.value;
        fire(inp, "input");
      });
      range.addEventListener("change", (e) => { e.stopPropagation(); fire(inp, "change"); });
    }
    const sync = () => {
      const v = parseFloat(inp.value);
      const off = !!inp.disabled;
      dec.disabled = inc.disabled = off;
      dec.classList.toggle("edge", !isNaN(v) && !isNaN(lo) && v <= lo);
      inc.classList.toggle("edge", !isNaN(v) && !isNaN(hi) && v >= hi);
      if (!range) return;
      range.disabled = off;
      const cur = isNaN(v) ? lo : Math.min(hi, Math.max(lo, v));
      if (range.value !== String(cur)) range.value = String(cur);
      range.style.setProperty("--p", (((cur - lo) / (hi - lo)) * 100).toFixed(2) + "%");
    };
    watchValue(inp, sync);
    if ("MutationObserver" in window) new MutationObserver(sync).observe(inp, { attributes: true, attributeFilter: ["disabled", "readonly"] });
    sync();
  }

  /* ---------- 多行文本框：字数 / 标签数 / 名单项数 ---------- */
  const LIST_IDS = new Set(["moderation_input_groups", "moderation_output_groups", "nsfw_group_blacklist", "draw_limit_admin_ids"]);
  const PROMPT_IDS = new Set(["default_positive", "default_negative", "img2img_default_positive", "img2img_default_negative", "promptEnhancerResult"]);
  function countText(ta) {
    const v = ta.value || "";
    if (LIST_IDS.has(ta.id)) {
      const n = v.split(/[\s,，;；、]+/).filter(Boolean).length;
      return n ? `${n} 项` : "空";
    }
    if (PROMPT_IDS.has(ta.id)) {
      const tags = v.split(/[,，\n]+/).map((s) => s.trim()).filter(Boolean).length;
      return tags ? `${tags} 个标签` : "空";
    }
    const len = [...v.replace(/\r/g, "")].length;
    return len ? `${len} 字` : "空";
  }
  function enhanceTextarea(ta) {
    if (done.has(ta)) return;
    const lab = ta.parentElement;
    if (!lab || lab.tagName !== "LABEL") return;
    done.add(ta);
    if (PROMPT_IDS.has(ta.id)) ta.classList.add("p5-mono");
    const cnt = make("span", "p5-cnt", { "aria-hidden": "true" });
    lab.classList.add("p5-has-cnt");
    lab.appendChild(cnt);
    const sync = () => {
      const t = countText(ta);
      if (cnt.textContent !== t) cnt.textContent = t;
      cnt.classList.toggle("zero", t === "空");
    };
    watchValue(ta, sync);
    sync();
  }

  /* ---------- 文件夹卡片：复制路径 ---------- */
  function copyText(str) {
    const fallback = () => {
      const t = make("textarea", "p5-clip", { readonly: "", "aria-hidden": "true" });
      t.value = str;
      document.body.appendChild(t);
      t.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
      t.remove();
      return ok;
    };
    try {
      if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(str).then(() => true, () => fallback());
      }
    } catch (_) {}
    return Promise.resolve(fallback());
  }
  function enhanceFolder(card) {
    if (done.has(card)) return;
    const code = card.querySelector("code[id^='path-']");
    if (!code) return;
    done.add(card);
    const btn = make("button", "p5-copy", { type: "button", title: "复制路径", "aria-label": "复制路径" });
    btn.innerHTML = icon("copy");
    code.after(btn);
    btn.addEventListener("click", () => {
      const path = (code.textContent || "").trim();
      if (!path || path === "未检测到") { tell("还没有检测到这个文件夹的路径"); return; }
      copyText(path).then((ok) => {
        if (ok) {
          btn.classList.add("ok");
          setTimeout(() => btn.classList.remove("ok"), 1200);
          tell("已复制路径：" + path);
        } else {
          tell("复制失败，请手动选中路径复制");
        }
      });
    });
  }

  /* ---------- 页签小红点：该页有未保存修改 ---------- */
  $$("[data-view-tab]").forEach((tab) => {
    if (tab.querySelector(".p5-tab-dot")) return;
    tab.appendChild(make("i", "p5-tab-dot", { "aria-hidden": "true", title: "这一页有未保存的修改" }));
  });

  /* ---------- 首次 + 动态渲染（LoRA 卡片等）时增强 ---------- */
  function enhance(scope) {
    if (scope.matches && scope.matches(NUM_SEL)) enhanceNumber(scope);
    $$(NUM_SEL, scope).forEach(enhanceNumber);
    $$("label > textarea.textarea", scope).forEach(enhanceTextarea);
    $$(".folder-card", scope).forEach(enhanceFolder);
  }
  enhance(main);
  if ("MutationObserver" in window) {
    let pending = new Set(), raf = 0;
    new MutationObserver((muts) => {
      for (const m of muts) for (const n of m.addedNodes) if (n.nodeType === 1 && !n.matches(".p5-num, .p5-range-wrap, .p5-cnt, .p5-copy")) pending.add(n);
      if (pending.size && !raf) {
        raf = requestAnimationFrame(() => {
          raf = 0;
          const items = [...pending];
          pending = new Set();
          items.forEach((n) => { if (n.isConnected) enhance(n); });
        });
      }
    }).observe(main, { childList: true, subtree: true });
  }
})();
