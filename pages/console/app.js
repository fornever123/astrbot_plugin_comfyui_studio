"use strict";

const API = "";
let config = {};
let modelData = {};
let loraItems = [];
let loraCategories = ["未分类"];
let loraCategoryFilter = "全部";
let loraSearchQuery = "";
let expandedLoraFile = "";
let loraSection = "normal";
let presetItems = {};
let editingPreset = "";
let artistPresetItems = {};
let editingArtistPreset = "";
let fallbackBridge = null;
let aiConnectionTested = false;
let builtInAnimaContextPreview = "";

// AstrBot 4.27+ 会注入官方桥接脚本；旧版页面、缓存页面或直接打开页面时，
// 可能暂时没有 window.AstrBotPluginPage。这里用同一套 postMessage 协议兜底，
// 避免整个控制台卡在“检查中”。
function createFallbackBridge() {
  if (fallbackBridge) return fallbackBridge;
  const pending = new Map();
  let requestId = 0;
  const parent = window.parent;
  const channel = "astrbot-plugin-page";

  window.addEventListener("message", event => {
    if (event.source !== parent || !event.data || event.data.channel !== channel) return;
    if (event.data.kind !== "response") return;
    const request = pending.get(event.data.requestId);
    if (!request) return;
    pending.delete(event.data.requestId);
    clearTimeout(request.timer);
    if (event.data.ok) request.resolve(event.data.data);
    else request.reject(new Error(event.data.error || "AstrBot 页面接口请求失败"));
  });

  const request = (action, payload, transfer = []) => new Promise((resolve, reject) => {
    if (parent === window) {
      reject(new Error("请从 AstrBot 插件控制台页面打开此页面"));
      return;
    }
    const id = `fallback_req_${++requestId}`;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error("AstrBot 页面桥接超时，请刷新插件页面"));
    }, 20000);
    pending.set(id, {resolve, reject, timer});
    parent.postMessage({channel, kind: "request", requestId: id, action, ...(payload || {})}, "*", transfer);
  });

  fallbackBridge = {
    ready: () => Promise.resolve(),
    apiGet: (endpoint, params) => request("api:get", {endpoint, params}),
    apiPost: (endpoint, body) => request("api:post", {endpoint, body}),
    upload: async (endpoint, file) => {
      if (!file || typeof file.arrayBuffer !== "function") throw new Error("没有选择要上传的文件");
      const fileBuffer = await file.arrayBuffer();
      return request("files:upload", {
        endpoint,
        fileName: file.name || "upload.bin",
        fileType: file.type || "application/octet-stream",
        fileLastModified: typeof file.lastModified === "number" ? file.lastModified : null,
        fileBuffer,
      }, [fileBuffer]);
    },
  };
  return fallbackBridge;
}

function pageBridge() {
  return window.AstrBotPluginPage || createFallbackBridge();
}

async function get(path) {
  const value = await pageBridge().apiGet(path.replace(/^\/+/, ""));
  if (value && value.error) throw new Error(value.error);
  return value;
}

async function post(path, body) {
  const value = await pageBridge().apiPost(path.replace(/^\/+/, ""), body);
  if (value && value.error) throw new Error(value.error);
  return value;
}

async function upload(path, file) {
  const value = await pageBridge().upload(path.replace(/^\/+/, ""), file);
  if (value && value.error) throw new Error(value.error);
  return value;
}

function show(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(show.timer);
  show.timer = setTimeout(() => el.classList.remove("show"), 3000);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
}

function externalUrl(value) {
  const url = String(value || "").trim();
  return /^https?:\/\//i.test(url) ? url : "";
}

async function copyText(value) {
  const text = String(value || "");
  if (!text) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {}
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  let copied = false;
  try { copied = document.execCommand("copy"); } catch (_) {}
  textarea.remove();
  return copied;
}

function setTheme(theme) {
  const root = document.documentElement;
  root.dataset.uiTheme = theme;
  root.classList.toggle("theme-light", theme === "light");
  root.classList.toggle("theme-dark", theme === "dark");
  // AstrBot 页面 iframe 使用 sandbox，部分浏览器会禁止访问 localStorage。
  // 主题保存失败不能阻断整个控制台初始化。
  try { localStorage.setItem("comfyui-ai-studio-theme", theme); } catch (_) {}
  const button = document.getElementById("themeToggle");
  if (button) button.textContent = theme === "light" ? "切换夜间" : "切换白天";
}

function initTheme() {
  let saved = "";
  try { saved = localStorage.getItem("comfyui-ai-studio-theme") || ""; } catch (_) {}
  setTheme(saved === "dark" ? "dark" : "light");
  const button = document.getElementById("themeToggle");
  if (button) button.onclick = () => setTheme(document.documentElement.dataset.uiTheme === "light" ? "dark" : "light");
}

function initLoraMode() {
  try {
    expandedLoraFile = localStorage.getItem("comfyui-ai-studio-expanded-lora")
      || localStorage.getItem("comfyui-ai-studio-expanded-style-lora")
      || "";
  } catch (_) {
    expandedLoraFile = "";
  }
}

function initLoraSection() {
  const apply = section => {
    loraSection = section === "style" ? "style" : "normal";
    const normalPanel = document.querySelector(".lora-management-band");
    const stylePanel = document.querySelector(".style-lora-band");
    if (normalPanel) normalPanel.hidden = loraSection !== "normal";
    if (stylePanel) stylePanel.hidden = loraSection !== "style";
    document.querySelectorAll("[data-lora-section]").forEach(tab => {
      const active = tab.dataset.loraSection === loraSection;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    try { localStorage.setItem("comfyui-ai-studio-lora-section", loraSection); } catch (_) {}
  };
  document.querySelectorAll("[data-lora-section]").forEach(tab => {
    tab.onclick = () => apply(tab.dataset.loraSection);
  });
  let saved = "";
  try { saved = localStorage.getItem("comfyui-ai-studio-lora-section") || ""; } catch (_) {}
  apply(saved || "normal");
}

function initViews() {
  const names = new Set([...document.querySelectorAll("[data-view-panel]")].map(panel => panel.dataset.viewPanel));
  const apply = name => {
    const selected = names.has(name) ? name : "workflow";
    document.querySelectorAll("[data-view-panel]").forEach(panel => { panel.hidden = panel.dataset.viewPanel !== selected; });
    document.querySelectorAll("[data-view-tab]").forEach(tab => {
      const active = tab.dataset.viewTab === selected;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    try { localStorage.setItem("comfyui-ai-studio-view", selected); } catch (_) {}
    if (window.location.hash !== `#${selected}`) history.replaceState(null, "", `#${selected}`);
  };
  document.querySelectorAll("[data-view-tab]").forEach(tab => {
    tab.onclick = () => apply(tab.dataset.viewTab);
  });
  window.addEventListener("hashchange", () => apply(window.location.hash.slice(1)));
  let saved = window.location.hash.slice(1);
  if (!saved) {
    try { saved = localStorage.getItem("comfyui-ai-studio-view") || ""; } catch (_) {}
  }
  apply(saved || "lora");
}

async function load() {
  initTheme();
  initLoraMode();
  initLoraSection();
  initViews();
  try {
    const status = await get(`${API}/status`);
    document.getElementById("version").textContent = `版本 ${status.version || "v0.10.3"}`;
    config = status.config || {};
    const comfy = status.comfy || {};
    const statusEl = document.getElementById("status");
    statusEl.textContent = comfy.ok ? "已连接" : "未连接";
    statusEl.className = `status ${comfy.ok ? "ok" : "bad"}`;
    document.getElementById("statusText").textContent = comfy.ok ? `ComfyUI 已连接，版本 ${comfy.version || "未知"}` : `ComfyUI 未连接：${comfy.error || "未知错误"}`;
    const paths = status.paths || {};
    const pathSources = status.path_sources || {};
    const sourceLabels = { override: "手动指定", extra: "额外路径", default: "默认位置" };
    const pathsEl = document.getElementById("paths");
    if (pathsEl) {
      const openable = ["diffusion_models", "checkpoints", "loras", "upscale_models", "unet", "text_encoders", "vae", "controlnet", "ipadapter", "clip_vision", "workflows", "source_workflow"];
      pathsEl.innerHTML = Object.entries(paths).map(([key, value]) => {
        const kind = key === "workflow_dir" ? "workflows" : key;
        const canOpen = value && openable.includes(kind);
        const val = value || "未检测到";
        const source = pathSources[key];
        const badge = value && source && source !== "default"
          ? `<span class="path-source">${escapeHtml(sourceLabels[source] || source)}</span>`
          : "";
        return `<div class="path-row"><code>${escapeHtml(key)}</code><span class="path-value" title="${escapeHtml(val)}">${escapeHtml(val)}</span>${badge}${canOpen ? `<button type="button" class="secondary" data-open-folder="${escapeHtml(kind)}">打开文件夹</button>` : ""}</div>`;
      }).join("");
      pathsEl.querySelectorAll("[data-open-folder]").forEach(btn => {
        btn.onclick = async () => {
          try {
            const result = await post(`${API}/open_folder`, { kind: btn.dataset.openFolder });
            show(`已打开：${result.path}`);
          } catch (e) { show(e.message); }
        };
      });
    }
    const pathsCountEl = document.getElementById("pathsCount");
    if (pathsCountEl) pathsCountEl.textContent = `（${Object.keys(paths).length} 项）`;
    for (const key of ["loras", "diffusion_models", "checkpoints", "upscale_models", "workflow_dir"]) {
      const el = document.getElementById(`path-${key}`);
      if (!el) continue;
      const value = paths[key];
      const source = pathSources[key];
      const label = value && source && source !== "default" ? sourceLabels[source] : "";
      el.textContent = label ? `${value} · ${label}` : (value || "未检测到");
    }
    const sourcePath = paths.source_workflow || "";
    const sourcePathElement = document.getElementById("path-source_workflow");
    if (sourcePathElement) {
      sourcePathElement.textContent = sourcePath ? sourcePath.replace(/[\\/][^\\/]+$/, "") : "未检测到";
    }
    fillConfig();
    await Promise.all([loadModels(), loadWorkflows(), loadPresets(), loadArtistPresets()]);
  } catch (error) {
    document.getElementById("status").textContent = "控制台接口异常";
    document.getElementById("status").className = "status bad";
    document.getElementById("statusText").textContent = `读取失败：${error.message || "未知错误"}`;
    show(`加载失败：${error.message || "未知错误"}`);
  }
}

function fillConfig() {
  document.getElementById("ai_base_url").value = config.ai_base_url || "";
  const civitaiBaseInput = document.getElementById("civitai_base_url");
  if (civitaiBaseInput) civitaiBaseInput.value = config.civitai_base_url || "https://civitai.com";
  document.getElementById("civitai_token").value = "";
  document.getElementById("civitai_token").placeholder = config.civitai_token_configured ? "已保存密钥，留空表示沿用" : "公共模型通常不需要";
  const civitaiStatus = document.getElementById("civitaiStatus");
  if (civitaiStatus) civitaiStatus.textContent = `当前信息源：${config.civitai_base_url || "https://civitai.com"}`;
  setAiModelOptions(config.ai_model ? [config.ai_model] : [], config.ai_model || "");
  document.getElementById("llm_prompt_source").value = config.llm_prompt_source || "astrbot";
  document.getElementById("plugin_ai_debug").checked = config.plugin_ai_debug === true || config.plugin_ai_debug === "true" || config.plugin_ai_debug === 1;
  document.getElementById("plugin_ai_llm_system_prompt").value = config.plugin_ai_llm_system_prompt || "";
  builtInAnimaContextPreview = String(config.plugin_ai_anima_context_preview || "");
  document.getElementById("plugin_ai_anima_context").value = config.plugin_ai_anima_context || builtInAnimaContextPreview;
  const animaStatus = document.getElementById("animaPluginStatus");
  if (animaStatus) {
    const info = config.anima_engineer_plugin || {};
    animaStatus.textContent = info.builtin ? `已启用内置功能：${info.template_path || "使用内置精简规则"}` : "内置 Anima 提示词工程师未加载";
    animaStatus.className = `muted ${info.builtin ? "success-text" : "error-text"}`;
  }
  document.getElementById("plugin_ai_command_system_prompt").value = config.plugin_ai_command_system_prompt || "";
  document.getElementById("img2img_llm_prompt_source").value = config.img2img_llm_prompt_source || "plugin";
  document.getElementById("img2img_plugin_ai_debug").checked = config.img2img_plugin_ai_debug === true || config.img2img_plugin_ai_debug === "true" || config.img2img_plugin_ai_debug === 1;
  const img2imgDefaults = config.img2img_prompt_defaults || {};
  document.getElementById("img2img_ai_system_prompt").value = config.img2img_ai_system_prompt || config.img2img_plugin_ai_llm_system_prompt || img2imgDefaults.system_prompt || img2imgDefaults.plugin_system_prompt || "";
  document.getElementById("img2img_plugin_ai_llm_system_prompt").value = config.img2img_plugin_ai_llm_system_prompt || "";
  document.getElementById("img2img_plugin_ai_knowledge").value = config.img2img_plugin_ai_knowledge || img2imgDefaults.knowledge || "";
  document.getElementById("img2img_astrbot_llm_system_prompt").value = config.img2img_astrbot_llm_system_prompt || img2imgDefaults.astrbot_system_prompt || "";
  document.getElementById("img2img_astrbot_user_prompt_template").value = config.img2img_astrbot_user_prompt_template || img2imgDefaults.astrbot_user_prompt_template || "";
  document.getElementById("img2img_plugin_ai_user_prompt_template").value = config.img2img_plugin_ai_user_prompt_template || img2imgDefaults.plugin_user_prompt_template || "";
  document.getElementById("img2img_plugin_ai_output_format").value = config.img2img_plugin_ai_output_format || img2imgDefaults.output_format || "";
  document.getElementById("img2img_llm_tool_prompt").value = config.img2img_llm_tool_prompt || img2imgDefaults.llm_tool_prompt || "";
  document.getElementById("plain_translate_enabled").checked = config.plain_translate_enabled !== false;
  document.getElementById("plain_translate_url").value = config.plain_translate_url || "https://translate.googleapis.com/translate_a/single";
  const img2imgTranslateEnabled = document.getElementById("img2img_plain_translate_enabled");
  if (img2imgTranslateEnabled) img2imgTranslateEnabled.checked = config.img2img_plain_translate_enabled === true || config.img2img_plain_translate_enabled === "true" || config.img2img_plain_translate_enabled === 1;
  const img2imgTranslateUrl = document.getElementById("img2img_plain_translate_url");
  if (img2imgTranslateUrl) img2imgTranslateUrl.value = config.img2img_plain_translate_url || "https://translate.googleapis.com/translate_a/single";
  document.getElementById("default_positive").value = config.default_positive || config.quality_prefix || "";
  document.getElementById("default_negative").value = config.default_negative || "";
  const img2imgEngine = document.getElementById("img2img_engine");
  if (img2imgEngine) img2imgEngine.value = config.img2img_engine || "qwen";
  document.getElementById("draw_reply_mode").value = "custom";
  document.getElementById("draw_delivery_mode").value = config.draw_delivery_mode || "normal";
  const llmStartMode = document.getElementById("llm_draw_start_reply_mode");
  if (llmStartMode) llmStartMode.value = config.llm_draw_start_reply_mode || "ai";
  const queueNoticeEnabled = document.getElementById("draw_queue_notice_enabled");
  if (queueNoticeEnabled) queueNoticeEnabled.checked = config.draw_queue_notice_enabled !== false && config.draw_queue_notice_enabled !== "false" && config.draw_queue_notice_enabled !== 0;
  const queueNoticeAi = document.getElementById("draw_queue_notice_ai");
  if (queueNoticeAi) queueNoticeAi.checked = config.draw_queue_notice_ai === true || config.draw_queue_notice_ai === "true" || config.draw_queue_notice_ai === 1;
  if (document.getElementById("llm_wait_timeout")) document.getElementById("llm_wait_timeout").value = config.llm_wait_timeout ?? 45;
  if (document.getElementById("draw_start_reply")) document.getElementById("draw_start_reply").value = config.draw_start_reply || "{mode}任务已提交，生成期间可以继续聊天，完成后会发送结果。";
  document.getElementById("draw_attach_prompt").checked = config.draw_attach_prompt === true || config.draw_attach_prompt === "true" || config.draw_attach_prompt === 1;
  document.getElementById("nsfw_group_blacklist").value = Array.isArray(config.nsfw_group_blacklist) ? config.nsfw_group_blacklist.join("\n") : (config.nsfw_group_blacklist || "");
  applyModerationConfig(config);
  document.getElementById("draw_reply_custom").value = config.draw_reply_custom || "{mode}完成，共 {count} 张。";
  const styleMode = document.getElementById("style_lora_mode");
  if (styleMode) styleMode.value = config.style_lora_mode || "random";
  const styleRandomCount = document.getElementById("style_lora_random_count");
  if (styleRandomCount) styleRandomCount.value = Math.max(1, Math.min(16, Number(config.style_lora_random_count || 1)));
  document.getElementById("draw_limit_count").value = config.draw_limit_count ?? 0;
  document.getElementById("draw_limit_window_seconds").value = config.draw_limit_window_seconds ?? 3600;
  const queueLimitEnabled = document.getElementById("draw_queue_limit_enabled");
  if (queueLimitEnabled) queueLimitEnabled.checked = config.draw_queue_limit_enabled === true || config.draw_queue_limit_enabled === "true" || config.draw_queue_limit_enabled === 1;
  const queueLimitCount = document.getElementById("draw_queue_limit_count");
  if (queueLimitCount) queueLimitCount.value = config.draw_queue_limit_count ?? 0;
  document.getElementById("draw_limit_admin_ids").value = Array.isArray(config.draw_limit_admin_ids) ? config.draw_limit_admin_ids.join("\n") : (config.draw_limit_admin_ids || "");
  document.getElementById("comfyui_start_script").value = config.comfyui_start_script || "";
  for (const id of ["width", "height", "steps", "seed", "cfg", "sampler_name", "scheduler", "denoise", "hires_scale", "hires_steps", "hires_denoise", "hires_upscale_model"]) {
    const element = document.getElementById(id);
    if (element) element.value = config[id] ?? "";
  }
  setSelectOptions("img2img_unet_name", modelData.img2img_models || [], config.img2img_unet_name || modelData.current_img2img || "");
  setSelectOptions("img2img_clip_name", modelData.img2img_clip_models || [], config.img2img_clip_name || modelData.current_img2img_clip || "");
  setSelectOptions("img2img_vae_name", modelData.img2img_vae_models || [], config.img2img_vae_name || modelData.current_img2img_vae || "");
  setSelectOptions("img2img_lora_name", modelData.loras || [], config.img2img_lora_name || "", true);
  setSelectOptions("img2img_accel_lora_name", modelData.img2img_accel_loras || modelData.loras || [], config.img2img_accel_lora_name || "", true);
  const img2imgAccelEnabled = document.getElementById("img2img_accel_lora_enabled");
  if (img2imgAccelEnabled) img2imgAccelEnabled.checked = config.img2img_accel_lora_enabled === true || config.img2img_accel_lora_enabled === "true" || config.img2img_accel_lora_enabled === 1;
  const img2imgPreCfg = document.getElementById("img2img_pre_cfg");
  if (img2imgPreCfg) img2imgPreCfg.checked = config.img2img_pre_cfg === true || config.img2img_pre_cfg === "true" || config.img2img_pre_cfg === 1;
  for (const id of ["img2img_lora_strength", "img2img_accel_lora_strength", "img2img_accel_steps", "img2img_accel_cfg", "img2img_width", "img2img_height", "img2img_steps", "img2img_cfg", "img2img_seed", "img2img_sampler_name", "img2img_scheduler", "img2img_denoise", "img2img_scale_method", "img2img_largest_size", "img2img_crop", "img2img_megapixels", "img2img_resolution_steps", "img2img_reference_method", "img2img_sampling_shift", "img2img_cfg_norm_strength", "img2img_tile_size", "img2img_tile_overlap", "img2img_temporal_size", "img2img_temporal_overlap", "img2img_default_positive", "img2img_default_negative", "img2img_second_image", "img2img_filename_prefix"]) {
    const element = document.getElementById(id);
    if (element) element.value = config[id] ?? "";
  }
  const matchInputSize = document.getElementById("img2img_match_input_size");
  if (matchInputSize) matchInputSize.checked = config.img2img_match_input_size !== false;
  const flux2TextFields = [
    "img2img_flux2_source_workflow", "img2img_flux2_lora_strength", "img2img_flux2_size",
    "img2img_flux2_steps", "img2img_flux2_cfg", "img2img_flux2_seed",
    "img2img_flux2_sampler_name", "img2img_flux2_scheduler", "img2img_flux2_denoise",
    "img2img_flux2_batch", "img2img_flux2_default_positive", "img2img_flux2_default_negative",
    "img2img_flux2_prompt_template", "img2img_flux2_ai_system_prompt",
    "img2img_flux2_plugin_ai_knowledge", "img2img_flux2_plain_translate_url",
    "img2img_flux2_output_format", "img2img_flux2_filename_prefix",
  ];
  for (const id of flux2TextFields) {
    const element = document.getElementById(id);
    if (element) element.value = config[id] ?? "";
  }
  const flux2Source = document.getElementById("img2img_flux2_llm_prompt_source");
  if (flux2Source) flux2Source.value = config.img2img_flux2_llm_prompt_source || "plugin";
  for (const id of ["img2img_flux2_plugin_ai_debug", "img2img_flux2_plain_translate_enabled"]) {
    const element = document.getElementById(id);
    if (element) element.checked = config[id] === true || config[id] === "true" || config[id] === 1;
  }
  for (const mode of ["wash", "outpaint", "multi_angle"]) {
    for (const key of ["source_workflow", "model_name", "clip_name", "vae_name", "default_positive", "default_negative", "sampler_name", "scheduler", "filename_prefix"]) {
      const element = document.getElementById(`${mode}_${key}`);
      if (element) element.value = config[`${mode}_${key}`] ?? "";
    }
    for (const key of ["steps", "cfg", "seed", "denoise", "size", "batch", "caption_tokens", "left", "top", "right", "bottom", "feathering", "horizontal_angle", "vertical_angle", "zoom"]) {
      const element = document.getElementById(`${mode}_${key}`);
      if (element) element.value = config[`${mode}_${key}`] ?? "";
    }
    for (const key of ["default_prompts", "camera_view"]) {
      const element = document.getElementById(`${mode}_${key}`);
      if (element) element.checked = config[`${mode}_${key}`] === true || config[`${mode}_${key}`] === "true" || config[`${mode}_${key}`] === 1;
    }
  }
  updateQwenAccelHint();
  updateReplyCustomVisibility();
}

function setSelectOptions(id, values, selected = "", allowEmpty = false) {
  const select = document.getElementById(id);
  if (!select) return;
  const options = [...new Set((values || []).map(value => String(value || "").trim()).filter(Boolean))];
  if (selected && !options.includes(selected)) options.unshift(selected);
  select.innerHTML = `${allowEmpty ? '<option value="">不使用</option>' : ''}${options.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  select.value = selected || (allowEmpty ? "" : options[0] || "");
}

function updateQwenAccelHint() {
  const hint = document.getElementById("img2imgAccelHint");
  if (!hint) return;
  const normalSteps = document.getElementById("img2img_steps")?.value || "8";
  const normalCfg = document.getElementById("img2img_cfg")?.value || "1.0";
  const accelSteps = document.getElementById("img2img_accel_steps")?.value || "4";
  const accelCfg = document.getElementById("img2img_accel_cfg")?.value || "1.0";
  const enabled = document.getElementById("img2img_accel_lora_enabled")?.checked === true;
  const model = document.getElementById("img2img_unet_name")?.value || "";
  const isRapid = model.toLowerCase().includes("rapid");
  const effectiveEnabled = enabled && !isRapid;
  const rapidWarning = isRapid
    ? " 当前核心为 Qwen-Rapid，插件会自动跳过标准 Lightning LoRA。"
    : "";
  hint.textContent = `普通模式（未启用加速 LoRA）：${normalSteps} 步 / CFG ${normalCfg}；加速模式（启用兼容 Lightning LoRA）：${accelSteps} 步 / CFG ${accelCfg}。当前生效：${effectiveEnabled ? `加速模式 ${accelSteps} 步 / CFG ${accelCfg}` : `普通模式 ${normalSteps} 步 / CFG ${normalCfg}`}。${rapidWarning}`;
}

function updateModelHint() {
  const hint = document.getElementById("modelHint");
  const model = document.getElementById("model")?.value || "";
  const profile = modelData.model_profiles?.[model];
  if (!hint) return;
  if (!profile) {
    hint.textContent = "";
    return;
  }
  const recommended = profile.recommended || {};
  const patch = profile.family === "Anima-2.9B"
    ? (modelData.anima_29b_patch_installed ? "专用节点已安装，重启 ComfyUI 后生效。" : "缺少专用节点，请安装 ComfyUI-Anima-2.9B 并重启 ComfyUI。")
    : "无需 2.9B 专用节点。";
  hint.textContent = `${profile.family}：${profile.description}。默认建议：${recommended.sampler_name || ""} / ${recommended.scheduler || ""} / ${recommended.steps || ""} 步 / CFG ${recommended.cfg || ""}。${patch}`;
}

function setAiModelOptions(models, selected = "") {
  const select = document.getElementById("ai_model");
  if (!select) return;
  const values = [...new Set((models || []).map(value => String(value || "").trim()).filter(Boolean))];
  if (selected && !values.includes(selected)) values.unshift(selected);
  select.innerHTML = values.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("") || '<option value="">请先获取模型列表</option>';
  select.value = selected || values[0] || "";
}

function aiFormBody(action) {
  return {
    action,
    ai_base_url: document.getElementById("ai_base_url").value.trim(),
    ai_api_key: document.getElementById("ai_api_key").value,
    ai_model: document.getElementById("ai_model").value.trim(),
  };
}

function updateReplyCustomVisibility() {
  document.getElementById("drawReplyCustomLabel").hidden = document.getElementById("draw_reply_mode").value !== "custom";
}

/* 图片安全审核：名单和密钥都在这里回填，密钥只回填“是否已配置”。 */
const MODERATION_TEXT_FIELDS = {
  moderation_input_groups: "",
  moderation_output_groups: "",
  moderation_base_url: "",
  moderation_model: "",
};
const MODERATION_NUMBER_FIELDS = {
  moderation_timeout: 30,
  moderation_max_side: 1024,
};

function moderationListText(value) {
  if (Array.isArray(value)) return value.join("\n");
  return value == null ? "" : String(value);
}

function applyModerationConfig(config) {
  for (const [id, fallback] of Object.entries(MODERATION_TEXT_FIELDS)) {
    const element = document.getElementById(id);
    if (element) element.value = id.endsWith("_groups") ? moderationListText(config[id]) : (config[id] ?? fallback);
  }
  for (const [id, fallback] of Object.entries(MODERATION_NUMBER_FIELDS)) {
    const element = document.getElementById(id);
    if (element) element.value = config[id] ?? fallback;
  }
  const strictness = document.getElementById("moderation_strictness");
  if (strictness) strictness.value = config.moderation_strictness || "standard";
  const enabled = document.getElementById("moderation_enabled");
  if (enabled) enabled.checked = config.moderation_enabled === true || config.moderation_enabled === "true" || config.moderation_enabled === 1;
  const failOpen = document.getElementById("moderation_fail_open");
  if (failOpen) failOpen.checked = config.moderation_fail_open !== false && config.moderation_fail_open !== "false" && config.moderation_fail_open !== 0;
  const status = document.getElementById("moderationStatus");
  if (status && config.moderation_api_key_configured) {
    status.textContent = config.moderation_ready ? "已配置审核模型" : "密钥已保存，模型或地址仍缺失";
  }
}

function moderationFormBody() {
  const body = { action: "test" };
  for (const id of Object.keys(MODERATION_TEXT_FIELDS)) {
    const element = document.getElementById(id);
    if (element) body[id] = element.value.trim();
  }
  for (const [id, fallback] of Object.entries(MODERATION_NUMBER_FIELDS)) {
    const element = document.getElementById(id);
    if (element) body[id] = Number(element.value || fallback);
  }
  const strictness = document.getElementById("moderation_strictness");
  if (strictness) body.moderation_strictness = strictness.value;
  const failOpen = document.getElementById("moderation_fail_open");
  if (failOpen) body.moderation_fail_open = failOpen.checked;
  const apiKey = document.getElementById("moderation_api_key");
  if (apiKey) body.moderation_api_key = apiKey.value;
  return body;
}

function renderLoraCategories() {
  const select = document.getElementById("loraCategoryFilter");
  if (!select) return;
  const values = ["全部", ...loraCategories.filter(value => value !== "全部")];
  select.innerHTML = values.map(value => `<option value="${escapeHtml(value)}">${value === "全部" ? "全部分类" : escapeHtml(value)}</option>`).join("");
  if (!values.includes(loraCategoryFilter)) loraCategoryFilter = "全部";
  select.value = loraCategoryFilter;
  renderLoraBatchCategories();
}

// 批量导入的分类下拉：只提供实际可用的分类（不含“全部”筛选项）。
function renderLoraBatchCategories() {
  const select = document.getElementById("loraBatchCategory");
  if (!select) return;
  const current = select.value;
  const values = loraCategories.filter(value => value && value !== "全部");
  select.innerHTML = `<option value="">未分类</option>${values.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  select.value = values.includes(current) ? current : "";
}

function normalizedLoraPresetEntries(item) {
  const result = [];
  const seen = new Set();
  const add = (tag, content = "") => {
    tag = String(tag || "").trim();
    content = String(content || "").trim();
    if (!tag && !content) return;
    const key = `${tag.toLowerCase()}\u0000${content.toLowerCase()}`;
    if (seen.has(key)) return;
    seen.add(key);
    result.push({tag, content});
  };
  (Array.isArray(item.lora_presets) ? item.lora_presets : []).forEach(entry => {
    if (entry && typeof entry === "object") add(entry.tag || entry.civitai_tag, entry.content || entry.value);
  });
  return result;
}

function groupedLoraPresetEntries(item) {
  const groups = [];
  const byContent = new Map();
  normalizedLoraPresetEntries(item).forEach(entry => {
    const key = entry.content.trim().toLowerCase();
    let group = byContent.get(key);
    if (!group) {
      group = {tags: [], content: entry.content};
      byContent.set(key, group);
      groups.push(group);
    }
    if (entry.tag && !group.tags.some(tag => tag.toLowerCase() === entry.tag.toLowerCase())) {
      group.tags.push(entry.tag);
    }
  });
  return groups;
}

function renderLoraPresetEditor(item, file, compact = false) {
  const groups = groupedLoraPresetEntries(item);
  const rows = groups.map(group => {
    const aliases = group.tags.map(tag => `<input class="input lora-preset-alias" data-lora-preset-alias value="${escapeHtml(tag)}" placeholder="指令简称">`).join("");
    return `<div class="lora-preset-row" data-lora-preset-row><div class="lora-preset-aliases"><div class="lora-preset-alias-list" data-lora-preset-alias-list>${aliases || `<input class="input lora-preset-alias" data-lora-preset-alias placeholder="指令简称">`}</div><button type="button" class="secondary lora-preset-add-alias" data-lora-action="add-lora-preset-alias" data-file="${file}" aria-label="添加同一预设的指令简称" title="添加同一预设的指令简称">+</button></div><textarea class="textarea lora-preset-content" data-lora-preset-content placeholder="这些指令简称对应的预设内容">${escapeHtml(group.content)}</textarea><button type="button" class="secondary" data-lora-action="remove-lora-preset" data-file="${file}" title="移除这一组预设">移除整组</button></div>`;
  }).join("");
  const triggerWords = Array.isArray(item.civitai_trigger_words) ? item.civitai_trigger_words.filter(Boolean) : (Array.isArray(item.trigger_words) ? item.trigger_words.filter(Boolean) : []);
  const triggerPanel = triggerWords.length
    ? `<div class="lora-trigger-preview"><strong>CivitAI 提示词（仅显示）</strong><code>${escapeHtml(triggerWords.join(", "))}</code><span class="muted">这里只展示 C 站 trainedWords，不会自动加入最终提示词</span></div>`
    : `<div class="lora-trigger-preview muted">CivitAI 未返回提示词；下方 LoRA 专属预设才会参与绘图</div>`;
  return `<div class="lora-preset-editor${compact ? " compact" : ""}"><div class="lora-preset-title"><strong>LoRA 专属预设</strong><span class="muted">左侧可添加多个指令简称，右侧内容会对应这一整组简称</span></div>${triggerPanel}<div class="lora-preset-list" data-lora-preset-list="${file}">${rows || `<span class="muted lora-preset-empty">暂无专属预设，点击下方按钮添加一组</span>`}</div><button type="button" class="secondary" data-lora-action="add-lora-preset" data-file="${file}">添加预设组</button></div>`;
}

function renderLorasRawLegacy() {
  const active = {};
  (config.lora_list || []).forEach(item => {
    const text = String(item);
    const split = text.lastIndexOf(":");
    const name = split > 0 ? text.slice(0, split) : text;
    active[name] = split > 0 ? text.slice(split + 1) : "0.8";
  });
  const allItems = loraItems.length ? loraItems : (modelData.loras || []).map(file_name => ({file_name, alias: file_name.replace(/\.[^.]+$/, ""), command_aliases: [], command_alias: "", category: "未分类", images: [], found: false, model_url: "", custom_url: "", custom_name: "", civitai_tags: [], lora_presets: [], show_images: true}));
  // 画风 LoRA 由下方黄色模块独立管理，普通区域不重复展示它们。
  const normalItems = allItems.filter(item => (item.category || "未分类") !== "画风");
  const search = loraSearchQuery.trim().toLowerCase();
  const categoryItems = loraCategoryFilter === "全部" ? normalItems : normalItems.filter(item => (item.category || "未分类") === loraCategoryFilter);
  const items = search ? categoryItems.filter(item => {
    const aliases = Array.isArray(item.command_aliases) ? item.command_aliases : (item.command_alias ? [item.command_alias] : []);
    return [item.file_name, item.alias, ...aliases].some(value => String(value || "").toLowerCase().includes(search));
  }) : categoryItems;
  const container = document.getElementById("loras");
  container.classList.toggle("lora-simple", loraSimpleMode);
  const toggle = document.getElementById("loraDisplayToggle");
  if (toggle) toggle.textContent = loraSimpleMode ? "完整管理" : "简洁选择";
  const bulkActions = document.getElementById("loraBulkActions");
  if (bulkActions) bulkActions.hidden = !loraSimpleMode;
  const renderCard = item => {
    const rawFile = String(item.file_name || "");
    const file = escapeHtml(rawFile);
    const commandAliases = Array.isArray(item.command_aliases) ? item.command_aliases : (item.command_alias ? [item.command_alias] : []);
    const commandAliasChips = commandAliases.map(alias => `<span class="command-alias-chip" data-command-alias-value="${escapeHtml(alias)}">${escapeHtml(alias)}<button type="button" class="command-alias-remove" data-lora-action="remove-command-alias" data-file="${file}" data-alias="${escapeHtml(alias)}" aria-label="删除简称 ${escapeHtml(alias)}" title="删除简称">×</button></span>`).join("");
    const category = item.category || "未分类";
    const isEnabled = active[rawFile] !== undefined;
    const url = externalUrl(item.model_url);
    const escapedUrl = escapeHtml(url);
    const customUrl = externalUrl(item.custom_url);
    const escapedCustomUrl = escapeHtml(customUrl);
    const customName = escapeHtml(item.custom_name || "");
    const images = item.show_images === false ? `<span class="muted">图片显示已关闭</span>` : (item.images || []).slice(0, 1).map(image => `<a class="civitai-image" href="${escapedUrl || "#"}" target="_blank" rel="noopener noreferrer" data-civitai-url="${escapedUrl}"><img src="${escapeHtml(image.url)}" loading="lazy" alt="${escapeHtml(item.alias)} 的 CivitAI 预览图"></a>`).join("");
    const linkText = item.custom_info ? "打开我的 CivitAI 信息" : (item.custom_link ? "打开我的 CivitAI 链接" : (item.found ? `打开 CivitAI：${escapeHtml(item.model_name)}` : "未匹配，打开 CivitAI 搜索"));
    const info = url ? `<div class="civitai-link-row"><a class="civitai-link ${item.found ? "" : "muted"}" href="${escapedUrl}" target="_blank" rel="noopener noreferrer" data-civitai-url="${escapedUrl}">${linkText}</a><button type="button" class="secondary" data-civitai-copy="${escapedUrl}">复制链接</button></div><code class="civitai-url">${escapedUrl}</code>` : `<span class="muted">暂未获得 CivitAI 链接</span>`;
    const categoryOptions = loraCategories.map(value => `<option value="${escapeHtml(value)}" ${value === category ? "selected" : ""}>${escapeHtml(value)}</option>`).join("");
    if (loraSimpleMode) {
      return `<article class="lora-card lora-card-simple"><label class="lora-check"><input type="checkbox" data-lora-checkbox="${file}" ${isEnabled ? "checked" : ""}><span>${escapeHtml(item.alias)} <small class="lora-state">${isEnabled ? "✅ 已开启" : "⬜ 未开启"}</small></span></label><code>${file}</code><div class="lora-simple-controls"><label>分类<select class="select lora-category-input" data-lora-category-file="${file}">${categoryOptions}</select></label>${renderLoraPresetEditor(item, file, true)}<button class="primary" data-lora-action="save-lora" data-file="${file}">保存分类与预设</button></div><small class="lora-simple-meta">指令简称：${escapeHtml(commandAliases.join("、") || "默认简称")}</small></article>`;
    }
    return `<article class="lora-card"><div class="lora-main"><label class="lora-check"><input type="checkbox" data-lora-checkbox="${file}" ${isEnabled ? "checked" : ""}><span>${escapeHtml(item.alias)} <small class="lora-state">${isEnabled ? "✅ 已开启" : "⬜ 未开启"}</small></span></label><code>${file}</code><div class="lora-controls"><label>分类<select class="select lora-category-input" data-lora-category-file="${file}">${categoryOptions}</select></label><label>显示昵称<input class="input alias-input" data-alias-file="${file}" value="${escapeHtml(item.alias)}"></label><label>权重<input class="weight" data-weight-file="${file}" type="number" min="0" max="2" step="0.05" value="${escapeHtml(active[item.file_name] || "0.8")}"></label><div class="command-alias-label"><span>指令简称（可添加多个）</span><div class="command-alias-list" data-command-alias-list="${file}">${commandAliasChips || `<span class="muted command-alias-empty">尚未添加简称，将使用默认简称</span>`}</div><div class="command-alias-add"><input class="input command-alias-input" data-command-alias-input-file="${file}" placeholder="如：1号lora"><button type="button" class="secondary" data-lora-action="add-command-alias" data-file="${file}">添加简称</button></div></div><label class="civitai-name-label">CivitAI 显示名称<input class="input" data-civitai-name-file="${file}" value="${customName}" placeholder="留空使用链接自动查询名称"></label><label class="civitai-url-label">我的 CivitAI 链接<input class="input civitai-url-input" data-civitai-url-file="${file}" value="${escapedCustomUrl}" placeholder="保存后立即更新图片和链接"></label><label class="check lora-show-images"><input type="checkbox" data-show-images-file="${file}" ${item.show_images !== false ? "checked" : ""}>显示 CivitAI 图片</label></div>${renderLoraPresetEditor(item, file)}<div class="lora-card-actions"><button class="primary" data-lora-action="save-lora" data-file="${file}">保存此 LoRA 全部设置</button><button class="secondary" data-lora-action="open-lora" data-file="${file}">打开 LoRA 文件位置</button></div></div><div class="civitai-info">${info}<div class="civitai-gallery">${images || `<span class="muted">暂无 CivitAI 图片</span>`}</div></div></article>`;
  };
  const grouped = new Map();
  items.forEach(item => {
    const category = item.category || "未分类";
    if (!grouped.has(category)) grouped.set(category, []);
    grouped.get(category).push(item);
  });
  const categoryOrder = new Map(loraCategories.map((value, index) => [value, index]));
  const modules = [...grouped.entries()]
    .sort((a, b) => (categoryOrder.get(a[0]) ?? 999) - (categoryOrder.get(b[0]) ?? 999))
    .map(([category, group]) => `<section class="lora-category-module" data-lora-category-module="${escapeHtml(category)}"><div class="lora-category-module-head"><div><h3>${escapeHtml(category)}</h3><span class="muted">${group.length} 个 LoRA</span></div><button type="button" class="secondary lora-category-jump" data-lora-category-jump="${escapeHtml(category)}">只看此分类</button></div><div class="lora-grid lora-category-grid">${group.map(renderCard).join("")}</div></section>`)
    .join("");
  container.innerHTML = modules || '<div class="empty">当前分类没有 LoRA。</div>';
}

function renderLorasLegacy() {
  renderLorasRaw();
  renderStyleLoras();
}

function styleLoraConfigLegacy() {
  const selected = new Set((config.style_lora_list || []).map(value => String(value || "").split(":")[0]));
  const weights = config.style_lora_weights && typeof config.style_lora_weights === "object" ? config.style_lora_weights : {};
  const aliases = config.style_lora_aliases && typeof config.style_lora_aliases === "object" ? config.style_lora_aliases : {};
  return {selected, weights, aliases};
}

function renderStyleLorasLegacy() {
  const container = document.getElementById("styleLoras");
  if (!container) return;
  const {selected, weights, aliases} = styleLoraConfig();
  const items = loraItems.length ? loraItems : (modelData.loras || []).map(file_name => ({file_name, alias: file_name.replace(/\.[^.]+$/, ""), category: "未分类"}));
  // 画风模块只展示普通 LoRA 管理中明确归类为“画风”的文件。
  // 旧版 style_lora_list 里残留的其它 LoRA 会被忽略，但不会从配置中删除。
  const styleCandidates = items.filter(item => (item.category || "未分类") === "画风");
  styleCandidates.sort((a, b) => {
    const aNumber = Number(String(a.style_alias || "").match(/^画风(\d+)$/)?.[1] || -1);
    const bNumber = Number(String(b.style_alias || "").match(/^画风(\d+)$/)?.[1] || -1);
    return bNumber - aNumber;
  });
  const styleItems = styleCandidates.filter(item => selected.has(item.file_name));
  const ordered = [...styleItems, ...styleCandidates.filter(item => !selected.has(item.file_name))];
  const active = currentLoraSelection();
  const renderCard = item => {
    const rawFile = String(item.file_name || "");
    const file = escapeHtml(rawFile);
    const checked = selected.has(rawFile);
    const styleIndex = styleCandidates.findIndex(value => value.file_name === rawFile);
    const rawAlias = aliases[rawFile] ?? aliases[rawFile.replace(/\\/g, "/")];
    const alias = Array.isArray(rawAlias) ? (rawAlias.find(value => String(value || "").trim()) || "") : (rawAlias || "");
    const styleAlias = String(alias || `画风${styleIndex + 1}`).trim();
    const weight = weights[rawFile] ?? weights[rawFile.replace(/\\/g, "/")] ?? 0.8;
    const isEnabled = active[rawFile] !== undefined;
    const category = item.category || "画风";
    const categoryOptions = loraCategories.map(value => `<option value="${escapeHtml(value)}" ${value === category ? "selected" : ""}>${escapeHtml(value)}</option>`).join("");
    const url = externalUrl(item.model_url);
    const escapedUrl = escapeHtml(url);
    const customUrl = externalUrl(item.custom_url);
    const customName = escapeHtml(item.custom_name || "");
    const images = item.show_images === false ? `<span class="muted">图片显示已关闭</span>` : (item.images || []).slice(0, 1).map(image => `<a class="civitai-image" href="${escapedUrl || "#"}" target="_blank" rel="noopener noreferrer" data-civitai-url="${escapedUrl}"><img src="${escapeHtml(image.url)}" loading="lazy" onerror="var p=this.closest('.lora-simple-thumb')||this.parentNode;p&amp;&amp;p.remove()" alt="${escapeHtml(item.alias || rawFile)} 的 CivitAI 预览图"></a>`).join("");
    const linkText = item.custom_info ? "打开我的 CivitAI 信息" : (item.custom_link ? "打开我的 CivitAI 链接" : (item.found ? `打开 CivitAI：${escapeHtml(item.model_name)}` : "未匹配，打开 CivitAI 搜索"));
    const info = url ? `<div class="civitai-link-row"><a class="civitai-link ${item.found ? "" : "muted"}" href="${escapedUrl}" target="_blank" rel="noopener noreferrer" data-civitai-url="${escapedUrl}">${linkText}</a><button type="button" class="secondary" data-civitai-copy="${escapedUrl}">复制链接</button></div><code class="civitai-url">${escapedUrl}</code>` : `<span class="muted">暂未获得 CivitAI 链接</span>`;
    const candidateCheck = `<label class="check style-lora-check"><input type="checkbox" data-style-lora-file="${file}" ${checked ? "checked" : ""}><span>纳入画风候选</span></label>`;
    const enabledCheck = `<label class="check style-lora-enabled"><input type="checkbox" data-lora-enabled-checkbox="${file}" ${isEnabled ? "checked" : ""}>普通 LoRA长期启用</label>`;
    const expanded = expandedStyleLoraFile === rawFile;
    if (!expanded) {
      return `<article class="style-lora-card style-lora-card-simple${checked ? " selected" : ""}"><div class="style-lora-card-head">${candidateCheck}<strong>${escapeHtml(item.alias || rawFile)}</strong><button class="secondary" type="button" data-lora-action="toggle-style-details" data-file="${file}">展开</button></div><code>${file}</code><label>详细昵称<input class="input" data-alias-file="${file}" value="${escapeHtml(item.alias || "")}"></label><label>指令简称<input class="input" data-style-lora-alias="${file}" value="${escapeHtml(styleAlias)}" placeholder="画风${styleIndex + 1}"></label><label>分类<select class="select lora-category-input" data-lora-category-file="${file}">${categoryOptions}</select></label><div class="style-lora-simple-actions">${enabledCheck}<button class="primary" data-lora-action="save-lora" data-file="${file}">保存此 LoRA</button></div></article>`;
    }
    const commandAliases = Array.isArray(item.command_aliases) ? item.command_aliases : (item.command_alias ? [item.command_alias] : []);
    const commandAliasChips = commandAliases.map(value => `<span class="command-alias-chip">${escapeHtml(value)}</span>`).join("");
    return `<article class="lora-card style-lora-card-full${checked ? " style-selected" : ""}"><div class="lora-main"><div class="style-lora-full-head"><div class="style-lora-check-row">${candidateCheck}${enabledCheck}</div><button class="secondary" type="button" data-lora-action="toggle-style-details" data-file="${file}">收起简洁选择</button></div><h3>${escapeHtml(item.alias || rawFile)}</h3><code>${file}</code><div class="lora-controls"><label>分类<select class="select lora-category-input" data-lora-category-file="${file}">${categoryOptions}</select></label><label>详细昵称<input class="input alias-input" data-alias-file="${file}" value="${escapeHtml(item.alias || "")}"></label><label>权重<input class="weight" data-style-lora-weight="${file}" data-weight-file="${file}" type="number" min="0" max="2" step="0.05" value="${escapeHtml(weight)}"></label><label>画风指令简称<input class="input" data-style-lora-alias="${file}" value="${escapeHtml(styleAlias)}" placeholder="画风${styleIndex + 1}"></label><div class="command-alias-label"><span>其它指令简称（只读显示）</span><div class="command-alias-list">${commandAliasChips || `<span class="muted">无</span>`}</div></div><label class="civitai-name-label">CivitAI 显示名称<input class="input" data-civitai-name-file="${file}" value="${customName}" placeholder="留空使用链接自动查询名称"></label><label class="civitai-url-label">我的 CivitAI 链接<input class="input civitai-url-input" data-civitai-url-file="${file}" value="${escapeHtml(customUrl)}" placeholder="保存后立即更新图片和链接"></label><label class="check lora-show-images"><input type="checkbox" data-show-images-file="${file}" ${item.show_images !== false ? "checked" : ""}>显示 CivitAI 图片</label></div>${renderLoraPresetEditor(item, file)}<div class="lora-card-actions"><button class="primary" data-lora-action="save-lora" data-file="${file}">保存此 LoRA 全部设置</button><button class="secondary" data-lora-action="open-lora" data-file="${file}">打开 LoRA 文件位置</button></div></div><div class="civitai-info">${info}<div class="civitai-gallery">${images || `<span class="muted">暂无 CivitAI 图片</span>`}</div></div></article>`;
  };
  container.innerHTML = ordered.length ? ordered.map(renderCard).join("") : '<div class="empty">当前没有分类为“画风”的 LoRA。请先在普通 LoRA管理中把目标文件分类为“画风”，或在本页上传并归类。</div>';
}

function loraManualAliases(item) {
  const values = Array.isArray(item.manual_command_aliases)
    ? item.manual_command_aliases
    : (Array.isArray(item.command_aliases) ? item.command_aliases : (item.command_alias ? [item.command_alias] : []));
  return [...new Set(values.map(value => String(value || "").trim()).filter(Boolean))];
}

function renderCommandAliasEditor(item, file) {
  const aliases = loraManualAliases(item);
  const chips = aliases.map(alias => `<span class="command-alias-chip" data-command-alias-value="${escapeHtml(alias)}">${escapeHtml(alias)}<button type="button" class="command-alias-remove" data-lora-action="remove-command-alias" data-file="${file}" data-alias="${escapeHtml(alias)}" aria-label="删除简称 ${escapeHtml(alias)}" title="删除简称">×</button></span>`).join("");
  return `<div class="command-alias-label"><span>指令简称（可添加多个）</span><div class="command-alias-list" data-command-alias-list="${file}">${chips || `<span class="muted command-alias-empty">尚未添加简称</span>`}</div><div class="command-alias-add"><input class="input command-alias-input" data-command-alias-input-file="${file}" placeholder="如：1号lora"><button type="button" class="secondary" data-lora-action="add-command-alias" data-file="${file}">添加简称</button></div></div>`;
}

function renderLoraCivitaiInfo(item) {
  const url = externalUrl(item.model_url);
  const escapedUrl = escapeHtml(url);
  const customName = escapeHtml(item.custom_name || "");
  const file = escapeHtml(item.file_name || "");
  const clearPreview = item.custom_image_data
    ? `<button type="button" class="secondary lora-preview-clear" data-lora-action="clear-lora-preview" data-file="${file}" title="清除拖入的预览图">清除预览图</button>`
    : "";
  const images = item.show_images === false
    ? `<span class="muted">图片显示已关闭</span>`
    : (item.images || []).slice(0, 1).map(image => `<a class="civitai-image" href="${escapedUrl || "#"}" target="_blank" rel="noopener noreferrer" data-civitai-url="${escapedUrl}"><img src="${escapeHtml(image.url)}" loading="lazy" alt="${escapeHtml(item.alias || item.file_name)} 的 CivitAI 预览图"></a>`).join("") + clearPreview;
  const linkText = item.custom_info
    ? "打开我的 CivitAI 信息"
    : (item.custom_link ? "打开我的 CivitAI 链接" : (item.found ? `打开 CivitAI：${escapeHtml(item.model_name)}` : "未匹配，打开 CivitAI 搜索"));
  const info = url
    ? `<div class="civitai-link-row"><a class="civitai-link ${item.found ? "" : "muted"}" href="${escapedUrl}" target="_blank" rel="noopener noreferrer" data-civitai-url="${escapedUrl}">${linkText}</a><button type="button" class="secondary" data-civitai-copy="${escapedUrl}">复制链接</button></div><code class="civitai-url">${escapedUrl}</code>`
    : `<span class="muted">暂未获得 CivitAI 链接</span>`;
  return {customName, images, info};
}

function renderLoraCompactCard(item, file, options = {}) {
  const style = Boolean(options.style);
  const aliases = Array.isArray(item.command_aliases)
    ? [...item.command_aliases]
    : (item.command_alias ? [item.command_alias] : []);
  if (item.style_alias) aliases.push(item.style_alias);
  const aliasText = [...new Set(aliases.map(value => String(value || "").trim()).filter(Boolean))].join("、") || "暂无简称";
  const civitai = renderLoraCivitaiInfo(item);
  const thumb = item.show_images === false || !(item.images || []).length
    ? ""
    : `<div class="lora-simple-thumb">${civitai.images}</div>`;
  return `<article class="lora-card lora-card-simple${style ? " style-lora-card-simple" : ""}${options.selected ? " selected" : ""}">${thumb}<div class="lora-simple-head"><strong title="${file}">${escapeHtml(item.alias || item.file_name)}</strong><span class="lora-simple-alias" title="指令简称">简称：${escapeHtml(aliasText)}</span><button class="secondary lora-card-toggle" type="button" data-lora-action="toggle-lora-details" data-file="${file}">展开</button></div></article>`;
}

function renderLoraFullCard(item, file, options = {}) {
  const style = Boolean(options.style);
  const category = item.category || (style ? "画风" : "未分类");
  const categoryOptions = loraCategories.map(value => `<option value="${escapeHtml(value)}" ${value === category ? "selected" : ""}>${escapeHtml(value)}</option>`).join("");
  const enabled = Boolean(options.enabled);
  const enabledInput = style
    ? `<label class="check"><input type="checkbox" data-lora-enabled-checkbox="${file}" ${enabled ? "checked" : ""}>长期启用</label>`
    : `<label class="lora-check"><input type="checkbox" data-lora-checkbox="${file}" ${enabled ? "checked" : ""}><span>${enabled ? "已开启" : "未开启"}</span></label>`;
  const candidate = style
    ? `<label class="check"><input type="checkbox" data-style-lora-file="${file}" ${options.selected ? "checked" : ""}>纳入随机候选</label>`
    : "";
  const styleAlias = style
    ? `<label>画风指令简称<input class="input" data-style-lora-alias="${file}" value="${escapeHtml(options.styleAlias || "")}" placeholder="画风${Number(options.styleIndex || 1)}"></label>`
    : "";
  const styleWeight = style
    ? `<label>随机画风权重<input class="weight" data-style-lora-weight="${file}" type="number" min="0" max="2" step="0.05" value="${escapeHtml(options.styleWeight ?? "0.8")}"></label>`
    : "";
  const info = renderLoraCivitaiInfo(item);
  return `<article class="lora-card${style ? " style-lora-card-full" : " lora-card-full"}"><div class="lora-main"><div class="lora-full-head"><div class="lora-check-row">${enabledInput}${candidate}</div><button class="secondary lora-card-toggle" type="button" data-lora-action="toggle-lora-details" data-file="${file}">收起简洁选择</button></div><h3>${escapeHtml(item.alias || item.file_name)}</h3><code>${file}</code><div class="lora-controls"><label>分类<select class="select lora-category-input" data-lora-category-file="${file}">${categoryOptions}</select></label><label>详细昵称<input class="input alias-input" data-alias-file="${file}" value="${escapeHtml(item.alias || "")}"></label><label>长期启用权重<input class="weight" data-weight-file="${file}" type="number" min="0" max="2" step="0.05" value="${escapeHtml(options.activeWeight ?? "0.8")}"></label>${styleWeight}${styleAlias}${renderCommandAliasEditor(item, file)}<label class="civitai-name-label">CivitAI 显示名称<input class="input" data-civitai-name-file="${file}" value="${info.customName}" placeholder="留空使用链接自动查询名称"></label><label class="civitai-url-label">我的 CivitAI 链接<input class="input civitai-url-input" data-civitai-url-file="${file}" value="${escapeHtml(externalUrl(item.custom_url))}" placeholder="保存后立即更新图片和链接"></label><label class="check lora-show-images"><input type="checkbox" data-show-images-file="${file}" ${item.show_images !== false ? "checked" : ""}>显示 CivitAI 图片</label></div>${renderLoraPresetEditor(item, file)}<div class="lora-card-actions"><button class="primary" data-lora-action="save-lora" data-file="${file}">保存此 LoRA 全部设置</button><button class="secondary" data-lora-action="open-lora" data-file="${file}">打开 LoRA 文件位置</button></div></div><div class="civitai-info">${info.info}<div class="civitai-gallery lora-image-dropzone" data-lora-image-dropzone="${file}" title="将图片拖入此区域替换预览图">${info.images || `<span class="muted">暂无 CivitAI 图片</span>`}</div></div></article>`;
}

function renderLorasRaw() {
  const active = currentLoraSelection();
  const allItems = loraItems.length ? loraItems : (modelData.loras || []).map(file_name => ({file_name, alias: file_name.replace(/\.[^.]+$/, ""), manual_command_aliases: [], command_aliases: [], category: "未分类", images: [], found: false, model_url: "", custom_url: "", custom_name: "", civitai_tags: [], lora_presets: [], show_images: true}));
  const normalItems = allItems.filter(item => (item.category || "未分类") !== "画风");
  const search = loraSearchQuery.trim().toLowerCase();
  const categoryItems = loraCategoryFilter === "全部" ? normalItems : normalItems.filter(item => (item.category || "未分类") === loraCategoryFilter);
  const items = search ? categoryItems.filter(item => [item.file_name, item.alias, ...loraManualAliases(item), item.style_alias].some(value => String(value || "").toLowerCase().includes(search))) : categoryItems;
  const container = document.getElementById("loras");
  const toggle = document.getElementById("loraDisplayToggle");
  if (toggle) toggle.textContent = "全部收起";
  const bulkActions = document.getElementById("loraBulkActions");
  if (bulkActions) bulkActions.hidden = false;
  const renderCard = item => {
    const rawFile = String(item.file_name || "");
    const file = escapeHtml(rawFile);
    const options = {enabled: active[rawFile] !== undefined, activeWeight: active[rawFile] || "0.8"};
    return expandedLoraFile === rawFile ? renderLoraFullCard(item, file, options) : renderLoraCompactCard(item, file, options);
  };
  const grouped = new Map();
  items.forEach(item => {
    const category = item.category || "未分类";
    if (!grouped.has(category)) grouped.set(category, []);
    grouped.get(category).push(item);
  });
  const categoryOrder = new Map(loraCategories.map((value, index) => [value, index]));
  const modules = [...grouped.entries()]
    .sort((a, b) => (categoryOrder.get(a[0]) ?? 999) - (categoryOrder.get(b[0]) ?? 999))
    .map(([category, group]) => `<section class="lora-category-module" data-lora-category-module="${escapeHtml(category)}"><div class="lora-category-module-head"><div><h3>${escapeHtml(category)}</h3><span class="muted">${group.length} 个 LoRA</span></div><button type="button" class="secondary lora-category-jump" data-lora-category-jump="${escapeHtml(category)}">只看此分类</button></div><div class="lora-grid lora-category-grid">${group.map(renderCard).join("")}</div></section>`)
    .join("");
  container.innerHTML = modules || '<div class="empty">当前分类没有 LoRA。</div>';
}

function renderLoras() {
  renderLorasRaw();
  renderStyleLoras();
}

function styleLoraConfig() {
  const selected = new Set((config.style_lora_list || []).map(value => String(value || "").split(":")[0]));
  const weights = config.style_lora_weights && typeof config.style_lora_weights === "object" ? config.style_lora_weights : {};
  const aliases = config.style_lora_aliases && typeof config.style_lora_aliases === "object" ? config.style_lora_aliases : {};
  return {selected, weights, aliases};
}

function renderStyleLoras() {
  const container = document.getElementById("styleLoras");
  if (!container) return;
  const {selected, weights, aliases} = styleLoraConfig();
  const items = loraItems.length ? loraItems : (modelData.loras || []).map(file_name => ({file_name, alias: file_name.replace(/\.[^.]+$/, ""), category: "未分类", manual_command_aliases: [], command_aliases: []}));
  const styleCandidates = items.filter(item => (item.category || "未分类") === "画风");
  styleCandidates.sort((a, b) => {
    const aNumber = Number(String(a.style_alias || "").match(/^画风(\d+)$/)?.[1] || -1);
    const bNumber = Number(String(b.style_alias || "").match(/^画风(\d+)$/)?.[1] || -1);
    return bNumber - aNumber;
  });
  const selectedKeys = new Set([...selected].map(value => String(value).replace(/\\/g, "/").toLowerCase()));
  const ordered = styleCandidates;
  const active = currentLoraSelection();
  const renderCard = item => {
    const rawFile = String(item.file_name || "");
    const file = escapeHtml(rawFile);
    const selectedItem = selectedKeys.has(rawFile.replace(/\\/g, "/").toLowerCase());
    const styleIndex = styleCandidates.findIndex(value => value.file_name === rawFile);
    const rawStyleAlias = aliases[rawFile] ?? aliases[rawFile.replace(/\\/g, "/")] ?? item.style_alias;
    const styleAlias = String(rawStyleAlias || `画风${styleIndex + 1}`).trim();
    const styleWeight = weights[rawFile] ?? weights[rawFile.replace(/\\/g, "/")] ?? item.style_weight ?? 0.8;
    const options = {style: true, selected: selectedItem, enabled: active[rawFile] !== undefined, activeWeight: active[rawFile] ?? "0.8", styleAlias, styleWeight, styleIndex: styleIndex + 1};
    return expandedLoraFile === rawFile ? renderLoraFullCard(item, file, options) : renderLoraCompactCard(item, file, options);
  };
  container.innerHTML = ordered.length ? ordered.map(renderCard).join("") : '<div class="empty">当前没有分类为“画风”的 LoRA。请先在 LoRA 管理中把目标文件分类为“画风”，或在本页上传并归类。</div>';
}

function readLoraPreviewData(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("读取图片失败，请重新选择"));
    reader.readAsDataURL(file);
  });
}

async function handleLoraPreviewDrop(event) {
  const zone = event.target.closest?.("[data-lora-image-dropzone]");
  if (!zone) return;
  event.preventDefault();
  event.stopPropagation();
  zone.classList.remove("dragover");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  if (!["image/png", "image/jpeg", "image/webp"].includes(String(file.type || "").toLowerCase())) {
    show("预览图只支持 PNG、JPEG 或 WEBP");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    show("预览图不能超过 8 MB");
    return;
  }
  try {
    const imageData = await readLoraPreviewData(file);
    const result = await post(`${API}/lora_info`, {
      action: "set_preview_image",
      file_name: zone.dataset.loraImageDropzone || "",
      image_data: imageData,
    });
    loraItems = result.items || loraItems;
    loraCategories = result.categories || loraCategories;
    renderLoraCategories();
    renderLoras();
    show("LoRA 预览图已更新");
  } catch (error) {
    show(error.message || "预览图更新失败");
  }
}

function bindLoraPreviewDrop(container) {
  if (!container) return;
  container.addEventListener("dragover", event => {
    const zone = event.target.closest?.("[data-lora-image-dropzone]");
    if (!zone || !container.contains(zone)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    zone.classList.add("dragover");
  });
  container.addEventListener("dragleave", event => {
    const zone = event.target.closest?.("[data-lora-image-dropzone]");
    if (!zone || zone.contains(event.relatedTarget)) return;
    zone.classList.remove("dragover");
  });
  container.addEventListener("drop", handleLoraPreviewDrop);
}

async function loadLoraInfo(force = false) {
  const value = force ? await post(`${API}/lora_info`, {action: "refresh"}) : await get(`${API}/lora_info`);
  loraItems = value.items || [];
  loraCategories = value.categories || ["未分类"];
  renderLoraCategories();
  renderLoras();
}

async function loadModels() {
  modelData = await get(`${API}/models`);
  const values = modelData.compatible_models || [...(modelData.diffusion_models || []), ...(modelData.checkpoints || [])];
  document.getElementById("model").innerHTML = values.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("") || '<option value="">未检测到模型</option>';
  document.getElementById("model").value = modelData.current_model || "";
  updateModelHint();
  setSelectOptions("img2img_unet_name", modelData.img2img_models || [], config.img2img_unet_name || modelData.current_img2img || "");
  setSelectOptions("img2img_clip_name", modelData.img2img_clip_models || [], config.img2img_clip_name || modelData.current_img2img_clip || "");
  setSelectOptions("img2img_vae_name", modelData.img2img_vae_models || [], config.img2img_vae_name || modelData.current_img2img_vae || "");
  setSelectOptions("img2img_lora_name", modelData.loras || [], config.img2img_lora_name || "", true);
  setSelectOptions("img2img_accel_lora_name", modelData.img2img_accel_loras || modelData.loras || [], config.img2img_accel_lora_name || "", true);
  setSelectOptions("img2img_flux2_unet_name", modelData.img2img_flux2_models || [], config.img2img_flux2_unet_name || modelData.current_img2img_flux2 || "");
  setSelectOptions("img2img_flux2_clip_name", modelData.img2img_flux2_clip_models || [], config.img2img_flux2_clip_name || modelData.current_img2img_flux2_clip || "");
  setSelectOptions("img2img_flux2_vae_name", modelData.img2img_flux2_vae_models || [], config.img2img_flux2_vae_name || modelData.current_img2img_flux2_vae || "");
  setSelectOptions("img2img_flux2_lora_name", modelData.img2img_flux2_loras || modelData.loras || [], config.img2img_flux2_lora_name || modelData.current_img2img_flux2_lora || "", true);
  await loadLoraInfo();
}

function currentLoraSelection() {
  const enabled = {};
  (config.lora_list || []).forEach(item => {
    const text = String(item);
    const split = text.lastIndexOf(":");
    const name = split > 0 ? text.slice(0, split) : text;
    enabled[name] = split > 0 ? text.slice(split + 1) : "0.8";
  });
  return enabled;
}

function currentLoraSelectionListWithout(fileName) {
  return (config.lora_list || []).filter(item => String(item).split(":")[0] !== fileName);
}

async function saveSimpleLoraSelection() {
  const selected = new Set([...document.querySelectorAll("[data-lora-checkbox]")].filter(input => input.checked).map(input => input.dataset.loraCheckbox));
  const current = currentLoraSelection();
  const normalItems = loraItems.filter(item => (item.category || "未分类") !== "画风");
  const visible = new Set((loraCategoryFilter === "全部" ? normalItems : normalItems.filter(item => (item.category || "未分类") === loraCategoryFilter)).map(item => item.file_name));
  const next = Object.entries(current).filter(([name]) => !visible.has(name) || selected.has(name)).map(([name, weight]) => `${name}:${weight}`);
  for (const name of selected) if (!current[name]) next.push(`${name}:0.8`);
  const result = await post(`${API}/config`, {lora_list: [...new Set(next)]});
  config.lora_list = result.lora_list || [...new Set(next)];
  renderLoras();
}

async function loadWorkflows() {
  const value = await get(`${API}/workflows`);
  const files = value.files || [];
  for (const id of ["wf_txt2img", "wf_img2img", "wf_img2img_flux2", "wf_hires", "wf_wash", "wf_outpaint", "wf_multi_angle"]) {
    const mode = id.substring(3);
    const select = document.getElementById(id);
    select.innerHTML = files.map(file => `<option value="${escapeHtml(file)}">${escapeHtml(file)}</option>`).join("") || '<option value="">无工作流</option>';
    select.value = (value.selected || {})[mode] || "";
  }
}

async function loadPresets() {
  const value = await get(`${API}/presets`);
  presetItems = value.presets || {};
  document.getElementById("presets").innerHTML = Object.entries(presetItems).map(([name, item]) => {
    const translated = String(item.translated || "").trim();
    const source = item._auto_source ? "旧版自动条目（已迁移前兼容保留）" : "可编辑预设";
    return `<div class="preset"><div><b>${escapeHtml(name)}</b><small>内容：${escapeHtml(item.content || "")}</small>${translated ? `<small>已保存翻译：${escapeHtml(translated)}</small>` : ""}<small class="muted">${source}</small></div><div class="button-row"><button data-preset-action="edit" data-name="${escapeHtml(name)}">编辑</button><button data-preset-action="remove" data-name="${escapeHtml(name)}">删除</button></div></div>`;
  }).join("") || '<div class="empty">当前没有预设</div>';
}

async function loadArtistPresets() {
  const value = await get(`${API}/artist_presets`);
  artistPresetItems = value.presets || {};
  const select = document.getElementById("artistPresetActive");
  select.innerHTML = `<option value="">不使用画师串</option>${Object.keys(artistPresetItems).map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")}`;
  select.value = value.active || "";
  document.getElementById("artistPresets").innerHTML = Object.entries(artistPresetItems).map(([name, item]) => `<div class="preset"><div><b>${escapeHtml(name)}${name === (value.active || "") ? " ✅ 当前" : ""}</b><small>${escapeHtml(item.content || "")}</small></div><div class="button-row"><button data-artist-preset-action="edit" data-name="${escapeHtml(name)}">编辑</button><button data-artist-preset-action="activate" data-name="${escapeHtml(name)}">启用</button><button data-artist-preset-action="remove" data-name="${escapeHtml(name)}">删除</button></div></div>`).join("") || '<div class="empty">当前没有画师串预设</div>';
}

document.querySelectorAll("[data-open-folder]").forEach(button => {
  button.onclick = async () => { try { const result = await post(`${API}/open_folder`, {kind: button.dataset.openFolder}); show(`已打开：${result.path}`); } catch (e) { show(e.message); } };
});
// ---- 批量导入 LoRA ----
// 通过逐文件上传实现批量导入：AstrBot 页面桥接一次只能携带一个文件，
// 顺序上传最稳，也便于给出「i/N」进度和逐文件失败明细。
function updateLoraBatchProgress(patch = {}) {
  const panel = document.getElementById("loraBatchProgress");
  if (!panel) return;
  panel.hidden = false;
  const stage = document.getElementById("loraBatchStage");
  const percent = document.getElementById("loraBatchPercent");
  const bar = document.getElementById("loraBatchBar");
  const file = document.getElementById("loraBatchFile");
  const result = document.getElementById("loraBatchResult");
  const failures = document.getElementById("loraBatchFailures");
  if (patch.stage != null) stage.textContent = patch.stage;
  if (patch.percent != null) {
    const value = Math.max(0, Math.min(100, Number(patch.percent) || 0));
    bar.value = value;
    percent.textContent = `${value.toFixed(value % 1 ? 1 : 0)}%`;
  }
  if (patch.file != null) file.textContent = patch.file;
  if (patch.result != null) result.textContent = patch.result;
  if (Array.isArray(patch.failures)) {
    failures.innerHTML = patch.failures
      .map(item => `<li><b>${escapeHtml(item.file_name || "未知文件")}</b>：${escapeHtml(item.error || "导入失败")}</li>`)
      .join("");
  }
}

async function importLoraBatch(fileList) {
  const list = Array.from(fileList || []).filter(Boolean);
  if (!list.length) return;
  const categorySelect = document.getElementById("loraBatchCategory");
  const category = categorySelect ? categorySelect.value : "";
  const saved = [];
  const failures = [];
  updateLoraBatchProgress({
    stage: `准备导入 ${list.length} 个文件`,
    percent: 0,
    file: "尚未开始",
    result: `0 成功 / 0 失败`,
    failures: [],
  });
  for (let index = 0; index < list.length; index += 1) {
    const file = list[index];
    updateLoraBatchProgress({
      stage: `正在导入 ${index + 1}/${list.length}`,
      file: file.name,
      percent: (index / list.length) * 100,
      result: `${saved.length} 成功 / ${failures.length} 失败`,
    });
    try {
      const result = await upload(`${API}/upload_lora`, file);
      const name = (result && result.file_name) || file.name;
      saved.push(name);
    } catch (e) {
      failures.push({file_name: file.name, error: e.message || "导入失败"});
    }
    updateLoraBatchProgress({
      percent: ((index + 1) / list.length) * 100,
      result: `${saved.length} 成功 / ${failures.length} 失败`,
      failures,
    });
  }
  // 统一归类：一次请求处理全部已导入文件。
  let categoryNote = "";
  if (category && saved.length) {
    try {
      await post(`${API}/lora_info`, {action: "category_batch", file_names: saved, category});
      categoryNote = `，已归入「${category}」分类`;
    } catch (e) {
      categoryNote = `，但归类失败：${e.message}`;
    }
  }
  await loadModels();
  if (category) await loadLoraInfo(true);
  updateLoraBatchProgress({
    stage: failures.length ? "导入完成（部分失败）" : "导入完成",
    percent: 100,
    file: saved.length ? `已导入：${saved.slice(0, 5).join("、")}${saved.length > 5 ? ` 等 ${saved.length} 个` : ""}` : "没有文件导入成功",
    result: `${saved.length} 成功 / ${failures.length} 失败`,
    failures,
  });
  show(`批量导入完成：${saved.length} 成功 / ${failures.length} 失败${categoryNote}`);
}

document.getElementById("loraFile").onchange = async event => {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  try { await importLoraBatch(files); } catch (e) { show(e.message); }
  finally { event.target.value = ""; }
};

document.getElementById("styleLoraFile").onchange = async event => {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  const failures = [];
  let ok = 0;
  try {
    for (let index = 0; index < files.length; index += 1) {
      try {
        await upload(`${API}/upload_style_lora`, files[index]);
        ok += 1;
      } catch (e) {
        failures.push({file_name: files[index].name, error: e.message || "导入失败"});
      }
    }
    await loadModels();
    loraSection = "style";
    document.querySelector("[data-lora-section='style']")?.click();
    show(`画风 LoRA 导入完成：${ok} 成功 / ${failures.length} 失败`);
  } catch (e) { show(e.message); }
  finally { event.target.value = ""; }
};
function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function updateLoraDownloadProgress(job) {
  const panel = document.getElementById("loraDownloadProgress");
  const bar = document.getElementById("loraDownloadBar");
  const percent = document.getElementById("loraDownloadPercent");
  const stage = document.getElementById("loraDownloadStage");
  const file = document.getElementById("loraDownloadFile");
  const bytes = document.getElementById("loraDownloadBytes");
  if (!panel || !job) return;
  panel.hidden = false;
  const hasTotal = Number(job.total || 0) > 0;
  const value = hasTotal ? Math.max(0, Math.min(100, Number(job.progress || 0))) : 0;
  bar.value = value;
  bar.removeAttribute("indeterminate");
  if (!hasTotal && job.status === "running") bar.classList.add("indeterminate");
  else bar.classList.remove("indeterminate");
  percent.textContent = hasTotal ? `${value.toFixed(value % 1 ? 1 : 0)}%` : "下载中";
  stage.textContent = job.stage || "正在处理";
  const mode = job.download_mode ? `（${job.download_mode}）` : "";
  file.textContent = job.file_name ? `文件：${job.file_name}${mode}` : (job.download_mode ? `正在读取模型信息（${job.download_mode}）` : "正在读取模型信息");
  bytes.textContent = hasTotal ? `${formatBytes(job.downloaded)} / ${formatBytes(job.total)}` : `${formatBytes(job.downloaded)} / 未知大小`;
}

function finishLoraDownloadProgress() {
  const button = document.getElementById("downloadLora");
  if (button) { button.disabled = false; button.textContent = "下载到 LoRA 文件夹"; }
}

async function waitForLoraDownload(jobId) {
  while (true) {
    const job = await post(`${API}/download_lora_progress`, {job_id: jobId});
    updateLoraDownloadProgress(job);
    if (job.status === "done") return job;
    if (job.status === "error") throw new Error(job.error || "LoRA 下载失败");
    await new Promise(resolve => setTimeout(resolve, 700));
  }
}

document.getElementById("downloadLora").onclick = async () => {
  const input = document.getElementById("civitaiDownloadUrl");
  const overwrite = document.getElementById("civitaiDownloadOverwrite").checked;
  const button = document.getElementById("downloadLora");
  const url = input.value.trim();
  if (!url) { show("请先输入 CivitAI 模型链接"); return; }
  try {
    button.disabled = true;
    button.textContent = "下载中…";
    updateLoraDownloadProgress({status: "queued", stage: "已创建下载任务", progress: 0, downloaded: 0, total: 0});
    show("LoRA 下载任务已创建");
    const task = await post(`${API}/download_lora`, {url, overwrite});
    if (!task.job_id) throw new Error("下载任务创建失败：未返回任务编号");
    const result = await waitForLoraDownload(task.job_id);
    input.value = "";
    document.getElementById("civitaiDownloadOverwrite").checked = false;
    await loadModels();
    await loadLoraInfo(true);
    await loadPresets();
    updateLoraDownloadProgress({...result, stage: "下载完成，模型列表已刷新"});
    show(`LoRA 下载完成：${result.file_name}`);
  } catch (e) {
    updateLoraDownloadProgress({status: "error", stage: "下载失败", error: e.message || "未知错误", progress: 0, downloaded: 0, total: 0});
    show(e.message);
  } finally { finishLoraDownloadProgress(); }
};
document.getElementById("uploadWorkflow").onclick = async () => {
  const input = document.getElementById("workflowFile");
  const file = input.files?.[0];
  if (!file) { show("请先选择工作流 JSON 文件"); return; }
  try {
    await upload(`${API}/upload_workflow`, file);
    await loadWorkflows();
    show("工作流上传成功，请选择并保存使用的工作流");
  } catch (e) { show(e.message); }
  finally { input.value = ""; }
};
document.getElementById("loraSearch").oninput = event => { loraSearchQuery = event.target.value || ""; renderLoras(); };
document.getElementById("loraCategoryFilter").onchange = event => { loraCategoryFilter = event.target.value; renderLoras(); };
document.getElementById("addLoraCategory").onclick = async () => {
  const input = document.getElementById("newLoraCategory");
  try {
    const result = await post(`${API}/lora_info`, {action: "category_add", category: input.value});
    loraCategories = result.categories || loraCategories;
    renderLoraCategories();
    input.value = "";
    show("自定义分类已添加");
  } catch (e) { show(e.message); }
};
document.getElementById("deleteLoraCategory").onclick = async () => {
  if (loraCategoryFilter === "全部" || loraCategoryFilter === "未分类") { show("请先选择要删除的自定义分类"); return; }
  try {
    const result = await post(`${API}/lora_info`, {action: "category_delete", category: loraCategoryFilter});
    loraCategories = result.categories || ["未分类"];
    loraCategoryFilter = "全部";
    renderLoraCategories();
    renderLoras();
    show("分类已删除，里面的 LoRA 已归入未分类");
  } catch (e) { show(e.message); }
};
document.getElementById("refreshLoras").onclick = async () => { try { await loadLoraInfo(true); await loadPresets(); show("CivitAI 信息和自动预设已刷新"); } catch (e) { show(e.message); } };
async function handleLoraClick(event) {
  const categoryJump = event.target.closest("[data-lora-category-jump]");
  if (categoryJump) {
    loraCategoryFilter = categoryJump.dataset.loraCategoryJump || "全部";
    const select = document.getElementById("loraCategoryFilter");
    if (select) select.value = loraCategoryFilter;
    renderLoras();
    return;
  }
  const copyButton = event.target.closest("button[data-civitai-copy]");
  if (copyButton) {
    const copied = await copyText(copyButton.dataset.civitaiCopy);
    show(copied ? "CivitAI 链接已复制，可在浏览器打开" : "复制失败，请手动复制下方链接");
    return;
  }
  const civitaiLink = event.target.closest("a[data-civitai-url]");
  if (civitaiLink) {
    const url = civitaiLink.dataset.civitaiUrl;
    event.preventDefault();
    const opened = window.open(url, "_blank", "noopener,noreferrer");
    if (!opened) {
      const copied = await copyText(url);
      show(copied ? "AstrBot 页面限制新窗口，链接已复制，可在浏览器打开" : "AstrBot 页面限制新窗口，请复制下方链接");
    }
    return;
  }
  const button = event.target.closest("button[data-lora-action]");
  if (!button) return;
  try {
    const file = button.dataset.file;
    if (button.dataset.loraAction === "toggle-style-details" || button.dataset.loraAction === "toggle-lora-details") {
      expandedLoraFile = expandedLoraFile === file ? "" : file;
      try { localStorage.setItem("comfyui-ai-studio-expanded-lora", expandedLoraFile); } catch (_) {}
      renderLoras();
      return;
    }
    if (button.dataset.loraAction === "add-lora-preset") {
      const list = document.querySelector(`[data-lora-preset-list="${CSS.escape(file)}"]`);
      if (!list) return;
      list.querySelector(".lora-preset-empty")?.remove();
      const row = document.createElement("div");
      row.className = "lora-preset-row";
      row.setAttribute("data-lora-preset-row", "");
      row.innerHTML = `<div class="lora-preset-aliases"><div class="lora-preset-alias-list" data-lora-preset-alias-list><input class="input lora-preset-alias" data-lora-preset-alias placeholder="指令简称"></div><button type="button" class="secondary lora-preset-add-alias" data-lora-action="add-lora-preset-alias" data-file="${escapeHtml(file)}" aria-label="添加同一预设的指令简称" title="添加同一预设的指令简称">+</button></div><textarea class="textarea lora-preset-content" data-lora-preset-content placeholder="这些指令简称对应的预设内容"></textarea><button type="button" class="secondary" data-lora-action="remove-lora-preset" data-file="${escapeHtml(file)}">移除整组</button>`;
      list.appendChild(row);
      row.querySelector("[data-lora-preset-alias]")?.focus();
      return;
    }
    if (button.dataset.loraAction === "add-lora-preset-alias") {
      const aliasList = button.closest("[data-lora-preset-row]")?.querySelector("[data-lora-preset-alias-list]");
      if (!aliasList) return;
      aliasList.insertAdjacentHTML("afterbegin", '<input class="input lora-preset-alias" data-lora-preset-alias placeholder="指令简称">');
      aliasList.querySelector("[data-lora-preset-alias]")?.focus();
      return;
    }
    if (button.dataset.loraAction === "remove-lora-preset") {
      const list = button.closest("[data-lora-preset-list]");
      button.closest("[data-lora-preset-row]")?.remove();
      if (list && !list.querySelector("[data-lora-preset-row]")) list.innerHTML = '<span class="muted lora-preset-empty">暂无专属预设，点击下方按钮添加一组</span>';
      return;
    }
    if (button.dataset.loraAction === "add-command-alias") {
      const input = document.querySelector(`[data-command-alias-input-file="${CSS.escape(file)}"]`);
      const list = document.querySelector(`[data-command-alias-list="${CSS.escape(file)}"]`);
      const value = input.value.trim();
      if (!value) { show("请输入要添加的指令简称"); return; }
      if (value.length > 40 || /[\s,:，：]/.test(value)) { show("指令简称不能超过 40 个字符，也不能包含空格、逗号或冒号"); return; }
      const exists = [...list.querySelectorAll("[data-command-alias-value]")].some(chip => chip.dataset.commandAliasValue.toLowerCase() === value.toLowerCase());
      if (exists) { show("这个指令简称已经添加"); return; }
      const empty = list.querySelector(".command-alias-empty");
      if (empty) empty.remove();
      const chip = document.createElement("span");
      chip.className = "command-alias-chip";
      chip.dataset.commandAliasValue = value;
      chip.textContent = value;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "command-alias-remove";
      remove.dataset.loraAction = "remove-command-alias";
      remove.dataset.file = file;
      remove.dataset.alias = value;
      remove.title = "删除简称";
      remove.setAttribute("aria-label", `删除简称 ${value}`);
      remove.textContent = "×";
      chip.appendChild(remove);
      list.appendChild(chip);
      input.value = "";
      show("简称已添加，请点击保存此 LoRA");
      return;
    }
    if (button.dataset.loraAction === "remove-command-alias") {
      const chip = button.closest("[data-command-alias-value]");
      const list = button.closest("[data-command-alias-list]");
      chip?.remove();
      if (list && !list.querySelector("[data-command-alias-value]")) list.innerHTML = '<span class="muted command-alias-empty">尚未添加简称，将使用默认简称</span>';
      show("简称已移除，请点击保存此 LoRA");
      return;
    }
    if (button.dataset.loraAction === "clear-lora-preview") {
      const result = await post(`${API}/lora_info`, {action: "clear_preview_image", file_name: file});
      loraItems = result.items || loraItems;
      loraCategories = result.categories || loraCategories;
      renderLoraCategories();
      renderLoras();
      show("LoRA 预览图已清除");
      return;
    }
    if (button.dataset.loraAction === "open-lora") {
      await post(API + "/open_lora", {file_name: file});
      show("已打开 LoRA 所在文件夹");
    } else if (button.dataset.loraAction === "delete-lora") {
      const item = loraItems.find(value => value.file_name === file);
      const name = item?.alias || file;
      if (!window.confirm(`确定删除 LoRA“${name}”吗？本地模型文件和该 LoRA 的管理信息都会删除。`)) return;
      const result = await post(`${API}/delete_lora`, {file_name: file});
      loraItems = loraItems.filter(value => value.file_name !== file);
      modelData.loras = (modelData.loras || []).filter(value => value !== file);
      config.lora_list = result.lora_list || currentLoraSelectionListWithout(file);
      await loadModels();
      await loadLoraInfo(true);
      await loadPresets();
      show(`LoRA 已删除：${name}`);
    } else if (button.dataset.loraAction === "save-lora") {
      const item = loraItems.find(value => value.file_name === file) || {};
      const aliasInput = document.querySelector(`[data-alias-file="${CSS.escape(file)}"]`);
      const commandList = document.querySelector(`[data-command-alias-list="${CSS.escape(file)}"]`);
      const categoryInput = document.querySelector(`[data-lora-category-file="${CSS.escape(file)}"]`);
      const weightInput = document.querySelector(`[data-weight-file="${CSS.escape(file)}"]`);
      const nameInput = document.querySelector(`[data-civitai-name-file="${CSS.escape(file)}"]`);
      const urlInput = document.querySelector(`[data-civitai-url-file="${CSS.escape(file)}"]`);
      const showImagesInput = document.querySelector(`[data-show-images-file="${CSS.escape(file)}"]`);
      const enabledInput = document.querySelector(`[data-lora-checkbox="${CSS.escape(file)}"]`);
      const styleSelectedInput = document.querySelector(`[data-style-lora-file="${CSS.escape(file)}"]`);
      const styleAliasInput = document.querySelector(`[data-style-lora-alias="${CSS.escape(file)}"]`);
      const presetList = document.querySelector(`[data-lora-preset-list="${CSS.escape(file)}"]`);
      const loraPresets = [...(presetList?.querySelectorAll("[data-lora-preset-row]") || [])].flatMap(row => {
        const content = row.querySelector("[data-lora-preset-content]")?.value.trim() || "";
        const aliases = [...row.querySelectorAll("[data-lora-preset-alias]")]
          .map(input => input.value.trim())
          .filter(Boolean);
        return aliases.length
          ? aliases.map(tag => ({tag, content}))
          : (content ? [{tag: "", content}] : []);
      });
      const result = await post(`${API}/lora_info`, {
        action: "save_lora",
        file_name: file,
        alias: aliasInput?.value ?? item.alias ?? "",
        command_aliases: commandList ? [...commandList.querySelectorAll("[data-command-alias-value]")].map(chip => chip.dataset.commandAliasValue) : (Array.isArray(item.command_aliases) ? item.command_aliases : []),
        category: categoryInput?.value || item.category || "未分类",
        weight: weightInput?.value || currentLoraSelection()[file] || "0.8",
        civitai_name: nameInput?.value ?? item.custom_name ?? "",
        civitai_url: urlInput?.value ?? item.custom_url ?? "",
        show_images: showImagesInput ? showImagesInput.checked : item.show_images !== false,
        enabled: enabledInput ? enabledInput.checked : (document.querySelector(`[data-lora-enabled-checkbox="${CSS.escape(file)}"]`)?.checked ?? currentLoraSelection()[file] !== undefined),
        ...(presetList ? {lora_presets: loraPresets} : {}),
        style_selected: styleSelectedInput ? styleSelectedInput.checked : undefined,
        style_alias: styleAliasInput?.value ?? undefined,
        style_weight: document.querySelector(`[data-style-lora-weight="${CSS.escape(file)}"]`)?.value ?? undefined,
      });
      loraItems = result.items || [];
      loraCategories = result.categories || loraCategories;
      config.lora_list = result.lora_list || config.lora_list || [];
      if (Array.isArray(result.style_lora_list)) config.style_lora_list = result.style_lora_list;
      if (result.style_lora_aliases && typeof result.style_lora_aliases === "object") config.style_lora_aliases = result.style_lora_aliases;
      if (result.style_lora_weights && typeof result.style_lora_weights === "object") config.style_lora_weights = result.style_lora_weights;
      renderLoraCategories();
      renderLoras();
      await loadPresets();
      show("此 LoRA 的全部设置已保存，CivitAI 图片已立即更新");
    }
  } catch (e) { show(e.message); }
};
document.getElementById("model").onchange = updateModelHint;
for (const id of ["img2img_accel_lora_enabled", "img2img_unet_name", "img2img_steps", "img2img_cfg", "img2img_accel_steps", "img2img_accel_cfg"]) {
  document.getElementById(id)?.addEventListener("input", updateQwenAccelHint);
  document.getElementById(id)?.addEventListener("change", updateQwenAccelHint);
}
document.getElementById("saveWorkflows").onclick = async () => { try { await post(`${API}/config`, {workflow_txt2img: document.getElementById("wf_txt2img").value, workflow_img2img: document.getElementById("wf_img2img").value, workflow_hires: document.getElementById("wf_hires").value}); show("工作流选择已保存"); } catch (e) { show(e.message); } };
document.getElementById("saveImg2ImgWorkflow").onclick = async () => { try { await post(`${API}/config`, {workflow_img2img: document.getElementById("wf_img2img").value}); show("图生图工作流已保存"); } catch (e) { show(e.message); } };
document.getElementById("saveImg2ImgFlux2Workflow").onclick = async () => {
  const payload = {
    img2img_engine: document.getElementById("img2img_engine").value,
    workflow_img2img_flux2: document.getElementById("wf_img2img_flux2").value,
    img2img_flux2_source_workflow: document.getElementById("img2img_flux2_source_workflow").value,
  };
  try { await post(`${API}/config`, payload); Object.assign(config, payload); show("Flux2 图生图工作流和默认引擎已保存"); }
  catch (e) { show(e.message); }
};
function flux2ToolPayload(mode) {
  const payload = {};
  for (const key of ["source_workflow", "model_name", "clip_name", "vae_name", "default_positive", "default_negative", "sampler_name", "scheduler", "filename_prefix"]) {
    payload[`${mode}_${key}`] = document.getElementById(`${mode}_${key}`).value;
  }
  for (const key of ["steps", "cfg", "seed", "denoise", "size", "batch", "caption_tokens", "left", "top", "right", "bottom", "feathering", "horizontal_angle", "vertical_angle", "zoom"]) {
    const element = document.getElementById(`${mode}_${key}`);
    if (element) payload[`${mode}_${key}`] = Number(element.value);
  }
  for (const key of ["default_prompts", "camera_view"]) {
    const element = document.getElementById(`${mode}_${key}`);
    if (element) payload[`${mode}_${key}`] = element.checked;
  }
  payload[`workflow_${mode}`] = document.getElementById(`wf_${mode}`).value;
  return payload;
}
for (const [id, mode, label] of [["saveWashSettings", "wash", "洗图"], ["saveOutpaintSettings", "outpaint", "扩图"], ["saveMultiAngleSettings", "multi_angle", "多角度"]]) {
  document.getElementById(id).onclick = async () => {
    try { const payload = flux2ToolPayload(mode); await post(`${API}/config`, payload); Object.assign(config, payload); show(`${label}设置已保存`); }
    catch (e) { show(e.message); }
  };
}
document.getElementById("saveTxt2ImgSettings").onclick = async () => {
  const payload = {};
  for (const id of ["width", "height", "steps", "seed", "cfg", "sampler_name", "scheduler", "denoise", "hires_scale", "hires_steps", "hires_denoise", "hires_upscale_model"]) {
    const value = document.getElementById(id).value;
    payload[id] = ["sampler_name", "scheduler", "hires_upscale_model"].includes(id) ? value : Number(value);
  }
  const animaInput = document.getElementById("plugin_ai_anima_context").value;
  payload.model_name = document.getElementById("model").value;
  payload.llm_prompt_source = document.getElementById("llm_prompt_source").value;
  payload.plugin_ai_debug = document.getElementById("plugin_ai_debug").checked;
  payload.plugin_ai_llm_system_prompt = document.getElementById("plugin_ai_llm_system_prompt").value;
  payload.plugin_ai_anima_context = animaInput.trim() === builtInAnimaContextPreview.trim() ? "" : animaInput;
  payload.plugin_ai_command_system_prompt = document.getElementById("plugin_ai_command_system_prompt").value;
  payload.plain_translate_enabled = document.getElementById("plain_translate_enabled").checked;
  payload.plain_translate_url = document.getElementById("plain_translate_url").value;
  payload.default_positive = document.getElementById("default_positive").value;
  payload.default_negative = document.getElementById("default_negative").value;
  try { await post(`${API}/config`, payload); Object.assign(config, payload); updateModelHint(); show("文生图模型、参数和提示词设置已保存"); } catch (e) { show(e.message); }
};
document.getElementById("loraDisplayToggle").onclick = () => {
  expandedLoraFile = "";
  try { localStorage.setItem("comfyui-ai-studio-expanded-lora", ""); } catch (_) {}
  renderLoras();
};
document.getElementById("loraSelectAll").onclick = () => {
  document.querySelectorAll("#loras [data-lora-checkbox]").forEach(input => { input.checked = true; });
};
document.getElementById("loraClearAll").onclick = () => {
  document.querySelectorAll("#loras [data-lora-checkbox]").forEach(input => { input.checked = false; });
};
document.getElementById("loraSaveSelection").onclick = async () => {
  try { await saveSimpleLoraSelection(); show("当前分类的 LoRA 选择已保存"); } catch (e) { show(e.message); }
};
document.getElementById("saveStyleLora").onclick = async () => {
  try {
    const selected = [...document.querySelectorAll("[data-style-lora-file]")]
      .filter(input => input.checked)
      .map(input => input.dataset.styleLoraFile);
    const aliases = config.style_lora_aliases && typeof config.style_lora_aliases === "object" ? {...config.style_lora_aliases} : {};
    const weights = config.style_lora_weights && typeof config.style_lora_weights === "object" ? {...config.style_lora_weights} : {};
    document.querySelectorAll("[data-style-lora-alias]").forEach(input => {
      const file = input.dataset.styleLoraAlias;
      const value = input.value.trim();
      if (value) aliases[file] = value;
    });
    document.querySelectorAll("[data-style-lora-weight]").forEach(input => {
      const file = input.dataset.styleLoraWeight;
      const value = Number(input.value);
      if (Number.isFinite(value)) weights[file] = Math.max(0, Math.min(2, value));
    });
    const payload = {
      style_lora_mode: document.getElementById("style_lora_mode").value,
      style_lora_random_count: Math.max(1, Math.min(16, Number(document.getElementById("style_lora_random_count")?.value || 1))),
      style_lora_list: selected,
      style_lora_aliases: aliases,
      style_lora_weights: weights,
    };
    const result = await post(`${API}/config`, payload);
    Object.assign(config, payload, result || {});
    renderStyleLoras();
    show("画风 LoRA 设置已保存，下一次文生图生效");
  } catch (e) { show(e.message); }
}
document.getElementById("loras").onclick = handleLoraClick;
document.getElementById("styleLoras").onclick = handleLoraClick;
bindLoraPreviewDrop(document.getElementById("loras"));
bindLoraPreviewDrop(document.getElementById("styleLoras"));
document.getElementById("loadAiModels").onclick = async () => {
  const status = document.getElementById("aiConnectionStatus");
  status.textContent = "正在获取模型列表…";
  try {
    const result = await post(`${API}/ai_models`, aiFormBody("list_models"));
    const models = result.models || [];
    const current = document.getElementById("ai_model").value;
    setAiModelOptions(models, models.includes(current) ? current : models[0] || "");
    status.textContent = `已获取 ${models.length} 个模型，请选择后测试连接`;
    status.className = "muted";
    show("模型列表获取成功");
  } catch (e) {
    status.textContent = `获取失败：${e.message}`;
    status.className = "muted error-text";
    show(e.message);
  }
};
document.getElementById("testAiConnection").onclick = async () => {
  const status = document.getElementById("aiConnectionStatus");
  status.textContent = "正在测试模型连接…";
  try {
    const result = await post(`${API}/ai_models`, aiFormBody("test"));
    aiConnectionTested = true;
    status.textContent = `连接成功：${result.model}`;
    status.className = "muted success-text";
    show("AI 连接测试成功");
  } catch (e) {
    aiConnectionTested = false;
    status.textContent = `测试失败：${e.message}`;
    status.className = "muted error-text";
    show(e.message);
  }
};
for (const id of ["ai_base_url", "ai_model", "ai_api_key", "civitai_base_url", "civitai_token"]) {
  const field = document.getElementById(id);
  if (!field) continue;
  field.addEventListener("input", () => { aiConnectionTested = false; });
  field.addEventListener("change", () => { aiConnectionTested = false; });
}
const saveCivitaiSettingsButton = document.getElementById("saveCivitaiSettings");
if (saveCivitaiSettingsButton) saveCivitaiSettingsButton.onclick = async () => {
  const status = document.getElementById("civitaiStatus");
  const payload = {
    civitai_base_url: (document.getElementById("civitai_base_url")?.value || config.civitai_base_url || "https://civitai.com").trim(),
    civitai_token: document.getElementById("civitai_token").value,
  };
  try {
    await post(`${API}/config`, payload);
    config.civitai_base_url = payload.civitai_base_url;
    if (payload.civitai_token) config.civitai_token_configured = true;
    if (status) {
      status.textContent = `已保存信息源：${payload.civitai_base_url}`;
      status.className = "muted success-text";
    }
    show("CivitAI 兼容站设置已保存");
  } catch (e) {
    if (status) {
      status.textContent = `保存失败：${e.message}`;
      status.className = "muted error-text";
    }
    show(e.message);
  }
};
document.getElementById("saveAi").onclick = async () => {
  if (!aiConnectionTested) {
    show("请先获取模型列表并测试连接，测试成功后再保存 AI 服务设置");
    return;
  }
  const payload = {
    ai_base_url: document.getElementById("ai_base_url").value,
    ai_model: document.getElementById("ai_model").value,
    ai_api_key: document.getElementById("ai_api_key").value,
    civitai_base_url: (document.getElementById("civitai_base_url")?.value || config.civitai_base_url || "https://civitai.com").trim(),
    civitai_token: document.getElementById("civitai_token").value,
  };
  try { await post(`${API}/config`, payload); Object.assign(config, payload); show("AI 服务设置已保存"); } catch (e) { show(e.message); }
};
document.getElementById("saveImg2ImgSettings").onclick = async () => {
  const payload = {};
  for (const id of ["img2img_unet_name", "img2img_clip_name", "img2img_vae_name", "img2img_lora_name", "img2img_accel_lora_name", "img2img_sampler_name", "img2img_scheduler", "img2img_scale_method", "img2img_default_positive", "img2img_default_negative", "img2img_filename_prefix"]) {
    payload[id] = document.getElementById(id).value;
  }
  for (const id of ["img2img_lora_strength", "img2img_accel_lora_strength", "img2img_accel_steps", "img2img_accel_cfg", "img2img_width", "img2img_height", "img2img_steps", "img2img_cfg", "img2img_seed", "img2img_denoise", "img2img_largest_size", "img2img_sampling_shift"]) {
    payload[id] = Number(document.getElementById(id).value);
  }
  payload.img2img_accel_lora_enabled = document.getElementById("img2img_accel_lora_enabled")?.checked === true;
  payload.img2img_keep_aspect_ratio = document.getElementById("img2img_keep_aspect_ratio")?.checked !== false;
  payload.img2img_match_input_size = document.getElementById("img2img_match_input_size")?.checked !== false;
  payload.img2img_crop = document.getElementById("img2img_crop").value;
  try {
    await post(`${API}/config`, payload);
    Object.assign(config, payload);
    show("独立图生图模型和节点参数已保存");
  } catch (e) { show(e.message); }
};
document.getElementById("saveImg2ImgAi").onclick = async () => {
  const payload = {
    img2img_llm_prompt_source: document.getElementById("img2img_llm_prompt_source").value,
    img2img_plugin_ai_debug: document.getElementById("img2img_plugin_ai_debug").checked,
    img2img_ai_system_prompt: document.getElementById("img2img_ai_system_prompt").value,
    img2img_plugin_ai_llm_system_prompt: "",
    img2img_plugin_ai_knowledge: document.getElementById("img2img_plugin_ai_knowledge").value,
    img2img_plain_translate_enabled: document.getElementById("img2img_plain_translate_enabled")?.checked === true,
    img2img_plain_translate_url: document.getElementById("img2img_plain_translate_url")?.value || "",
  };
  try { await post(`${API}/config`, payload); Object.assign(config, payload); show("图生图 LLM 和插件 AI 设置已保存"); } catch (e) { show(e.message); }
};
document.getElementById("saveImg2ImgFlux2Settings").onclick = async () => {
  const payload = {};
  for (const id of ["img2img_flux2_unet_name", "img2img_flux2_clip_name", "img2img_flux2_vae_name", "img2img_flux2_lora_name", "img2img_flux2_sampler_name", "img2img_flux2_scheduler", "img2img_flux2_default_positive", "img2img_flux2_default_negative", "img2img_flux2_filename_prefix"]) {
    payload[id] = document.getElementById(id).value;
  }
  for (const id of ["img2img_flux2_lora_strength", "img2img_flux2_size", "img2img_flux2_steps", "img2img_flux2_cfg", "img2img_flux2_seed", "img2img_flux2_denoise", "img2img_flux2_batch"]) {
    payload[id] = Number(document.getElementById(id).value);
  }
  try { await post(`${API}/config`, payload); Object.assign(config, payload); show("Flux2 图生图模型和参数已保存"); }
  catch (e) { show(e.message); }
};
document.getElementById("saveImg2ImgFlux2Ai").onclick = async () => {
  const payload = {
    img2img_flux2_llm_prompt_source: document.getElementById("img2img_flux2_llm_prompt_source").value,
    img2img_flux2_plugin_ai_debug: document.getElementById("img2img_flux2_plugin_ai_debug").checked,
    img2img_flux2_plain_translate_enabled: document.getElementById("img2img_flux2_plain_translate_enabled").checked,
    img2img_flux2_plain_translate_url: document.getElementById("img2img_flux2_plain_translate_url").value,
    img2img_flux2_ai_system_prompt: document.getElementById("img2img_flux2_ai_system_prompt").value,
    img2img_flux2_plugin_ai_knowledge: document.getElementById("img2img_flux2_plugin_ai_knowledge").value,
    img2img_flux2_prompt_template: document.getElementById("img2img_flux2_prompt_template").value,
    img2img_flux2_output_format: document.getElementById("img2img_flux2_output_format").value,
  };
  try { await post(`${API}/config`, payload); Object.assign(config, payload); show("Flux2 中文 AI 设置已保存"); }
  catch (e) { show(e.message); }
};
document.getElementById("saveDrawLimit").onclick = async () => {
  const payload = {
    draw_limit_count: Number(document.getElementById("draw_limit_count").value || 0),
    draw_limit_window_seconds: Number(document.getElementById("draw_limit_window_seconds").value || 3600),
    draw_queue_limit_enabled: document.getElementById("draw_queue_limit_enabled")?.checked === true,
    draw_queue_limit_count: Number(document.getElementById("draw_queue_limit_count")?.value || 0),
    draw_limit_admin_ids: document.getElementById("draw_limit_admin_ids").value,
    comfyui_start_script: document.getElementById("comfyui_start_script").value,
  };
  try { await post(`${API}/config`, payload); Object.assign(config, payload); show("绘图限额与 ComfyUI 控制设置已保存"); } catch (e) { show(e.message); }
};
document.getElementById("draw_reply_mode").onchange = updateReplyCustomVisibility;
document.getElementById("testModerationConnection").onclick = async () => {
  const status = document.getElementById("moderationStatus");
  status.textContent = "正在测试审核链路…";
  status.className = "muted";
  try {
    const result = await post(`${API}/moderation_test`, moderationFormBody());
    status.textContent = `审核链路正常（尺度：${result.strictness}）`;
    status.className = "muted success-text";
    show(result.message || "审核链路测试成功");
  } catch (e) {
    status.textContent = `测试失败：${e.message}`;
    status.className = "muted error-text";
    show(e.message);
  }
};
document.getElementById("saveModeration").onclick = async () => {
  const payload = {
    moderation_enabled: document.getElementById("moderation_enabled").checked,
    moderation_input_groups: document.getElementById("moderation_input_groups").value,
    moderation_output_groups: document.getElementById("moderation_output_groups").value,
    moderation_base_url: document.getElementById("moderation_base_url").value.trim(),
    moderation_api_key: document.getElementById("moderation_api_key").value,
    moderation_model: document.getElementById("moderation_model").value.trim(),
    moderation_strictness: document.getElementById("moderation_strictness").value,
    moderation_timeout: Number(document.getElementById("moderation_timeout").value || 30),
    moderation_max_side: Number(document.getElementById("moderation_max_side").value || 1024),
    moderation_fail_open: document.getElementById("moderation_fail_open").checked,
  };
  if (payload.moderation_enabled) {
    if (!payload.moderation_base_url || !payload.moderation_model) {
      show("启用审核前请先填写审核服务地址和审核模型；也可以先点“测试审核链路”验证");
      return;
    }
    if (!payload.moderation_input_groups.trim() && !payload.moderation_output_groups.trim()) {
      show("启用审核后，输入端和输出端名单至少要填写一个，否则不会检测任何会话");
      return;
    }
    if (!payload.moderation_api_key && !config.moderation_api_key_configured) {
      show("请填写审核 API Key");
      return;
    }
  }
  try {
    await post(`${API}/config`, payload);
    Object.assign(config, payload);
    if (payload.moderation_api_key) config.moderation_api_key_configured = true;
    config.moderation_ready = Boolean(payload.moderation_base_url && payload.moderation_model);
    document.getElementById("moderation_api_key").value = "";
    applyModerationConfig(config);
    show("图片安全审核设置已保存");
  } catch (e) { show(e.message); }
};
document.getElementById("saveReply").onclick = async () => { try { const payload = {draw_reply_mode: "custom", draw_delivery_mode: document.getElementById("draw_delivery_mode").value, llm_draw_start_reply_mode: document.getElementById("llm_draw_start_reply_mode")?.value || "ai", draw_queue_notice_enabled: document.getElementById("draw_queue_notice_enabled")?.checked !== false, draw_queue_notice_ai: document.getElementById("draw_queue_notice_ai")?.checked === true, llm_wait_timeout: Number(document.getElementById("llm_wait_timeout")?.value || 45), draw_attach_prompt: document.getElementById("draw_attach_prompt").checked, nsfw_group_blacklist: document.getElementById("nsfw_group_blacklist").value, draw_start_reply: document.getElementById("draw_start_reply")?.value || "", draw_reply_custom: document.getElementById("draw_reply_custom").value}; await post(`${API}/config`, payload); Object.assign(config, payload); show("回复、排队提示和群聊黑名单已保存"); } catch (e) { show(e.message); } };
document.getElementById("addPreset").onclick = async () => {
  const name = document.getElementById("presetName").value;
  const content = document.getElementById("presetContent").value;
  const wasEditing = !!editingPreset;
  try { await post(`${API}/presets`, {action: wasEditing ? "update" : "add", old_name: editingPreset, name, content}); resetPresetEditor(); await loadPresets(); show(wasEditing ? "预设已修改" : "预设已添加"); } catch (e) { show(e.message); }
};
document.getElementById("cancelPreset").onclick = resetPresetEditor;
document.getElementById("presets").onclick = async event => {
  const button = event.target.closest("button[data-preset-action]");
  if (!button) return;
  const action = button.dataset.presetAction;
  const name = button.dataset.name;
  try {
    if (action === "edit") {
      editingPreset = name;
      document.getElementById("presetName").value = name;
      document.getElementById("presetContent").value = presetItems[name]?.content || "";
      document.getElementById("addPreset").textContent = "保存修改";
      document.getElementById("cancelPreset").hidden = false;
      document.getElementById("presetName").focus();
      return;
    }
    await post(`${API}/presets`, {action, name});
    await loadPresets();
    show("预设已删除");
  } catch (e) { show(e.message); }
};

function resetPresetEditor() {
  editingPreset = "";
  document.getElementById("presetName").value = "";
  document.getElementById("presetContent").value = "";
  document.getElementById("addPreset").textContent = "添加预设";
  document.getElementById("cancelPreset").hidden = true;
}

document.getElementById("activateArtistPreset").onclick = async () => {
  try {
    await post(`${API}/artist_presets`, {action: "activate", name: document.getElementById("artistPresetActive").value});
    await loadArtistPresets();
    show("当前画师串已启用");
  } catch (e) { show(e.message); }
};
document.getElementById("disableArtistPreset").onclick = async () => {
  try {
    await post(`${API}/artist_presets`, {action: "activate", name: ""});
    await loadArtistPresets();
    show("画师串已关闭");
  } catch (e) { show(e.message); }
};
document.getElementById("addArtistPreset").onclick = async () => {
  const name = document.getElementById("artistPresetName").value;
  const content = document.getElementById("artistPresetContent").value;
  const wasEditing = !!editingArtistPreset;
  try {
    await post(`${API}/artist_presets`, {action: wasEditing ? "update" : "add", name, content});
    resetArtistPresetEditor();
    await loadArtistPresets();
    show(wasEditing ? "画师串预设已修改" : "画师串预设已添加");
  } catch (e) { show(e.message); }
};
document.getElementById("cancelArtistPreset").onclick = resetArtistPresetEditor;
document.getElementById("artistPresets").onclick = async event => {
  const button = event.target.closest("button[data-artist-preset-action]");
  if (!button) return;
  const action = button.dataset.artistPresetAction;
  const name = button.dataset.name;
  try {
    if (action === "edit") {
      editingArtistPreset = name;
      document.getElementById("artistPresetName").value = name;
      document.getElementById("artistPresetContent").value = artistPresetItems[name]?.content || "";
      document.getElementById("addArtistPreset").textContent = "保存修改";
      document.getElementById("cancelArtistPreset").hidden = false;
      document.getElementById("artistPresetName").focus();
      return;
    }
    if (action === "activate") await post(`${API}/artist_presets`, {action: "activate", name});
    if (action === "remove") await post(`${API}/artist_presets`, {action: "remove", name});
    await loadArtistPresets();
    show(action === "remove" ? "画师串预设已删除" : "画师串预设已启用");
  } catch (e) { show(e.message); }
};

function resetArtistPresetEditor() {
  editingArtistPreset = "";
  document.getElementById("artistPresetName").value = "";
  document.getElementById("artistPresetContent").value = "";
  document.getElementById("addArtistPreset").textContent = "添加画师串";
  document.getElementById("cancelArtistPreset").hidden = true;
}

load();


/* 未保存修改提示：作用域内任何表单变化 → 保存按钮变深（dirty） */
(function () {
  "use strict";
  function track(btnId) {
    var btn = document.getElementById(btnId);
    if (!btn) return;
    var band = btn.closest(".band") || btn.closest("section") || document;
    function mark() {
      if (!btn.classList.contains("dirty")) btn.classList.add("dirty");
    }
    band.addEventListener("input", mark, true);
    band.addEventListener("change", mark, true);
    btn.addEventListener("click", function () {
      setTimeout(function () { btn.classList.remove("dirty"); }, 400);
    });
  }
  ["loraSaveSelection", "saveStyleLora", "saveWorkflows", "saveTxt2ImgSettings",
   "saveImg2ImgWorkflow", "saveImg2ImgSettings", "saveImg2ImgAi",
   "saveImg2ImgFlux2Workflow", "saveImg2ImgFlux2Settings", "saveImg2ImgFlux2Ai",
   "saveWashSettings", "saveOutpaintSettings", "saveMultiAngleSettings",
   "saveAi", "saveReply", "saveDrawLimit", "saveModeration"].forEach(track);
})();


/* 动态卡片（展开的 LoRA 卡）保存按钮 dirty：事件委托覆盖动态渲染的卡片 */
(function () {
  "use strict";
  function saveBtn(card) {
    return card ? card.querySelector('[data-lora-action="save-lora"]') : null;
  }
  function mark(card) {
    var btn = saveBtn(card);
    if (btn && !btn.classList.contains("dirty")) btn.classList.add("dirty");
  }
  function cardOf(e) {
    var t = e && e.target;
    return t && t.closest ? t.closest(".lora-card-full, .style-lora-card-full") : null;
  }
  document.addEventListener("input", function (e) { mark(cardOf(e)); }, true);
  document.addEventListener("change", function (e) { mark(cardOf(e)); }, true);
  document.addEventListener("click", function (e) {
    var btn = e && e.target && e.target.closest
      ? e.target.closest('[data-lora-action="save-lora"]')
      : null;
    if (btn) setTimeout(function () { btn.classList.remove("dirty"); }, 400);
  }, true);
})();

