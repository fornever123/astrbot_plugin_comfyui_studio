"use strict";

const API = "";
let config = {};
let modelData = {};
let loraItems = [];
let loraCategories = ["未分类"];
let loraCategoryFilter = "全部";
let loraSimpleMode = false;
let presetItems = {};
let editingPreset = "";
let artistPresetItems = {};
let editingArtistPreset = "";
let fallbackBridge = null;
let aiConnectionTested = false;

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
  setTheme(saved === "light" ? "light" : "dark");
  const button = document.getElementById("themeToggle");
  if (button) button.onclick = () => setTheme(document.documentElement.dataset.uiTheme === "light" ? "dark" : "light");
}

function initLoraMode() {
  try {
    loraSimpleMode = localStorage.getItem("comfyui-ai-studio-lora-simple") === "1";
  } catch (_) {
    loraSimpleMode = false;
  }
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
  initViews();
  try {
    const status = await get(`${API}/status`);
    document.getElementById("version").textContent = `版本 ${status.version || "v0.6.4"}`;
    config = status.config || {};
    const comfy = status.comfy || {};
    const statusEl = document.getElementById("status");
    statusEl.textContent = comfy.ok ? "已连接" : "未连接";
    statusEl.className = `status ${comfy.ok ? "ok" : "bad"}`;
    document.getElementById("statusText").textContent = comfy.ok ? `ComfyUI 已连接，版本 ${comfy.version || "未知"}` : `ComfyUI 未连接：${comfy.error || "未知错误"}`;
    const paths = status.paths || {};
    document.getElementById("paths").textContent = Object.entries(paths).map(([key, value]) => `${key}: ${value || "未检测到"}`).join("\n");
    for (const key of ["loras", "diffusion_models", "checkpoints", "upscale_models", "workflow_dir"]) {
      const el = document.getElementById(`path-${key}`);
      if (el) el.textContent = paths[key] || "未检测到";
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
  setAiModelOptions(config.ai_model ? [config.ai_model] : [], config.ai_model || "");
  document.getElementById("plain_translate_enabled").checked = config.plain_translate_enabled !== false;
  document.getElementById("plain_translate_url").value = config.plain_translate_url || "https://translate.googleapis.com/translate_a/single";
  document.getElementById("default_positive").value = config.default_positive || config.quality_prefix || "";
  document.getElementById("default_negative").value = config.default_negative || "";
  document.getElementById("draw_reply_mode").value = config.draw_reply_mode || "astrbot";
  document.getElementById("draw_delivery_mode").value = config.draw_delivery_mode || "normal";
  document.getElementById("draw_reply_timeout").value = config.draw_reply_timeout ?? 6;
  document.getElementById("draw_start_reply").value = config.draw_start_reply || "{mode}任务已提交，生成期间可以继续聊天，完成后会发送结果。";
  document.getElementById("draw_reply_custom").value = config.draw_reply_custom || "{mode}完成，共 {count} 张。";
  for (const id of ["width", "height", "steps", "seed", "cfg", "sampler_name", "scheduler", "denoise", "hires_scale", "hires_steps", "hires_denoise", "hires_upscale_model"]) {
    const element = document.getElementById(id);
    if (element) element.value = config[id] ?? "";
  }
  updateReplyCustomVisibility();
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

function renderLoraCategories() {
  const select = document.getElementById("loraCategoryFilter");
  if (!select) return;
  const values = ["全部", ...loraCategories.filter(value => value !== "全部")];
  select.innerHTML = values.map(value => `<option value="${escapeHtml(value)}">${value === "全部" ? "全部分类" : escapeHtml(value)}</option>`).join("");
  if (!values.includes(loraCategoryFilter)) loraCategoryFilter = "全部";
  select.value = loraCategoryFilter;
}

function renderLoras() {
  const active = {};
  (config.lora_list || []).forEach(item => {
    const text = String(item);
    const split = text.lastIndexOf(":");
    const name = split > 0 ? text.slice(0, split) : text;
    active[name] = split > 0 ? text.slice(split + 1) : "0.8";
  });
  const allItems = loraItems.length ? loraItems : (modelData.loras || []).map(file_name => ({file_name, alias: file_name.replace(/\.[^.]+$/, ""), command_aliases: [], command_alias: "", category: "未分类", images: [], found: false, model_url: "", custom_url: "", custom_name: "", show_images: true}));
  const items = loraCategoryFilter === "全部" ? allItems : allItems.filter(item => (item.category || "未分类") === loraCategoryFilter);
  const container = document.getElementById("loras");
  container.classList.toggle("lora-simple", loraSimpleMode);
  const toggle = document.getElementById("loraDisplayToggle");
  if (toggle) toggle.textContent = loraSimpleMode ? "完整管理" : "简洁选择";
  const bulkActions = document.getElementById("loraBulkActions");
  if (bulkActions) bulkActions.hidden = !loraSimpleMode;
  container.innerHTML = items.map(item => {
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
      return `<article class="lora-card lora-card-simple"><label class="lora-check"><input type="checkbox" data-lora-checkbox="${file}" ${isEnabled ? "checked" : ""}><span>${escapeHtml(item.alias)} <small class="lora-state">${isEnabled ? "✅ 已开启" : "⬜ 未开启"}</small></span></label><code>${file}</code><small class="lora-simple-meta">分类：${escapeHtml(category)}　简称：${escapeHtml(commandAliases.join("、") || "默认简称")}</small></article>`;
    }
    return `<article class="lora-card"><div class="lora-main"><label class="lora-check"><input type="checkbox" data-lora-checkbox="${file}" ${isEnabled ? "checked" : ""}><span>${escapeHtml(item.alias)} <small class="lora-state">${isEnabled ? "✅ 已开启" : "⬜ 未开启"}</small></span></label><code>${file}</code><div class="lora-controls"><label>分类<select class="select lora-category-input" data-lora-category-file="${file}">${categoryOptions}</select></label><label>显示昵称<input class="input alias-input" data-alias-file="${file}" value="${escapeHtml(item.alias)}"></label><label>权重<input class="weight" data-weight-file="${file}" type="number" min="0" max="2" step="0.05" value="${escapeHtml(active[item.file_name] || "0.8")}"></label><div class="command-alias-label"><span>指令简称（可添加多个）</span><div class="command-alias-list" data-command-alias-list="${file}">${commandAliasChips || `<span class="muted command-alias-empty">尚未添加简称，将使用默认简称</span>`}</div><div class="command-alias-add"><input class="input command-alias-input" data-command-alias-input-file="${file}" placeholder="如：1号lora"><button type="button" class="secondary" data-lora-action="add-command-alias" data-file="${file}">添加简称</button></div></div><label class="civitai-name-label">CivitAI 显示名称<input class="input" data-civitai-name-file="${file}" value="${customName}" placeholder="留空使用链接自动查询名称"></label><label class="civitai-url-label">我的 CivitAI 链接<input class="input civitai-url-input" data-civitai-url-file="${file}" value="${escapedCustomUrl}" placeholder="保存后立即更新图片和链接"></label><label class="check lora-show-images"><input type="checkbox" data-show-images-file="${file}" ${item.show_images !== false ? "checked" : ""}>显示 CivitAI 图片</label><div class="lora-card-actions"><button class="primary" data-lora-action="save-lora" data-file="${file}">保存此 LoRA 全部设置</button><button class="danger-button" data-lora-action="delete-lora" data-file="${file}">删除此 LoRA</button></div></div></div><div class="civitai-info">${info}<div class="civitai-gallery">${images || `<span class="muted">暂无 CivitAI 图片</span>`}</div></div></article>`;
  }).join("") || '<div class="empty">当前分类没有 LoRA。</div>';
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
  const visible = new Set((loraCategoryFilter === "全部" ? loraItems : loraItems.filter(item => (item.category || "未分类") === loraCategoryFilter)).map(item => item.file_name));
  const next = Object.entries(current).filter(([name]) => !visible.has(name) || selected.has(name)).map(([name, weight]) => `${name}:${weight}`);
  for (const name of selected) if (!current[name]) next.push(`${name}:0.8`);
  const result = await post(`${API}/config`, {lora_list: [...new Set(next)]});
  config.lora_list = result.lora_list || [...new Set(next)];
  renderLoras();
}

async function loadWorkflows() {
  const value = await get(`${API}/workflows`);
  const files = value.files || [];
  for (const id of ["wf_txt2img", "wf_img2img", "wf_hires"]) {
    const mode = id.substring(3);
    const select = document.getElementById(id);
    select.innerHTML = files.map(file => `<option value="${escapeHtml(file)}">${escapeHtml(file)}</option>`).join("") || '<option value="">无工作流</option>';
    select.value = (value.selected || {})[mode] || "";
  }
}

async function loadPresets() {
  const value = await get(`${API}/presets`);
  presetItems = value.presets || {};
  document.getElementById("presets").innerHTML = Object.entries(presetItems).map(([name, item]) => `<div class="preset"><div><b>${escapeHtml(name)}</b><small>内容：${escapeHtml(item.content || "")}</small></div><div class="button-row"><button data-preset-action="edit" data-name="${escapeHtml(name)}">编辑</button><button data-preset-action="remove" data-name="${escapeHtml(name)}">删除</button></div></div>`).join("") || '<div class="empty">当前没有预设</div>';
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
document.getElementById("loraFile").onchange = async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  try { await upload(`${API}/upload_lora`, file); show("LoRA 已上传，正在刷新模型列表"); await loadModels(); } catch (e) { show(e.message); } finally { event.target.value = ""; }
};
document.getElementById("downloadLora").onclick = async () => {
  const input = document.getElementById("civitaiDownloadUrl");
  const overwrite = document.getElementById("civitaiDownloadOverwrite").checked;
  const url = input.value.trim();
  if (!url) { show("请先输入 CivitAI 模型链接"); return; }
  try {
    show("正在下载 LoRA，请稍候…");
    const result = await post(`${API}/download_lora`, {url, overwrite});
    input.value = "";
    document.getElementById("civitaiDownloadOverwrite").checked = false;
    await loadModels();
    show(`LoRA 下载完成：${result.file_name}`);
  } catch (e) { show(e.message); }
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
document.getElementById("loras").onclick = async event => {
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
    if (button.dataset.loraAction === "delete-lora") {
      const item = loraItems.find(value => value.file_name === file);
      const name = item?.alias || file;
      if (!window.confirm(`确定删除 LoRA“${name}”吗？本地模型文件和该 LoRA 的管理信息都会删除。`)) return;
      const result = await post(`${API}/delete_lora`, {file_name: file});
      loraItems = loraItems.filter(value => value.file_name !== file);
      modelData.loras = (modelData.loras || []).filter(value => value !== file);
      config.lora_list = result.lora_list || currentLoraSelectionListWithout(file);
      renderLoras();
      await loadPresets();
      show(`LoRA 已删除：${name}`);
    } else if (button.dataset.loraAction === "save-lora") {
      const aliasInput = document.querySelector(`[data-alias-file="${CSS.escape(file)}"]`);
      const commandList = document.querySelector(`[data-command-alias-list="${CSS.escape(file)}"]`);
      const categoryInput = document.querySelector(`[data-lora-category-file="${CSS.escape(file)}"]`);
      const weightInput = document.querySelector(`[data-weight-file="${CSS.escape(file)}"]`);
      const nameInput = document.querySelector(`[data-civitai-name-file="${CSS.escape(file)}"]`);
      const urlInput = document.querySelector(`[data-civitai-url-file="${CSS.escape(file)}"]`);
      const showImagesInput = document.querySelector(`[data-show-images-file="${CSS.escape(file)}"]`);
      const enabledInput = document.querySelector(`[data-lora-checkbox="${CSS.escape(file)}"]`);
      const result = await post(`${API}/lora_info`, {
        action: "save_lora",
        file_name: file,
        alias: aliasInput.value,
        command_aliases: [...commandList.querySelectorAll("[data-command-alias-value]")].map(chip => chip.dataset.commandAliasValue),
        category: categoryInput.value,
        weight: weightInput.value,
        civitai_name: nameInput.value,
        civitai_url: urlInput.value,
        show_images: showImagesInput.checked,
        enabled: enabledInput.checked,
      });
      loraItems = result.items || [];
      loraCategories = result.categories || loraCategories;
      config.lora_list = result.lora_list || config.lora_list || [];
      renderLoraCategories();
      renderLoras();
      await loadPresets();
      show("此 LoRA 的全部设置已保存，CivitAI 图片已立即更新");
    }
  } catch (e) { show(e.message); }
};
document.getElementById("saveModel").onclick = async () => { try { await post(`${API}/config`, {model_name: document.getElementById("model").value}); show("模型已保存"); } catch (e) { show(e.message); } };
document.getElementById("saveWorkflows").onclick = async () => { try { await post(`${API}/config`, {workflow_txt2img: document.getElementById("wf_txt2img").value, workflow_img2img: document.getElementById("wf_img2img").value, workflow_hires: document.getElementById("wf_hires").value}); show("工作流选择已保存"); } catch (e) { show(e.message); } };
document.getElementById("saveDrawSettings").onclick = async () => {
  const payload = {};
  for (const id of ["width", "height", "steps", "seed", "cfg", "sampler_name", "scheduler", "denoise", "hires_scale", "hires_steps", "hires_denoise", "hires_upscale_model"]) {
    const value = document.getElementById(id).value;
    payload[id] = ["sampler_name", "scheduler", "hires_upscale_model"].includes(id) ? value : Number(value);
  }
  try { await post(`${API}/config`, payload); Object.assign(config, payload); show("绘画参数已保存"); } catch (e) { show(e.message); }
};
document.getElementById("loraDisplayToggle").onclick = () => {
  loraSimpleMode = !loraSimpleMode;
  try { localStorage.setItem("comfyui-ai-studio-lora-simple", loraSimpleMode ? "1" : "0"); } catch (_) {}
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
for (const id of ["ai_base_url", "ai_model", "ai_api_key"]) {
  document.getElementById(id).addEventListener("input", () => { aiConnectionTested = false; });
}
document.getElementById("saveAi").onclick = async () => {
  try { await post(`${API}/config`, {ai_base_url: document.getElementById("ai_base_url").value, ai_model: document.getElementById("ai_model").value, ai_api_key: document.getElementById("ai_api_key").value, plain_translate_enabled: document.getElementById("plain_translate_enabled").checked, plain_translate_url: document.getElementById("plain_translate_url").value, default_positive: document.getElementById("default_positive").value, default_negative: document.getElementById("default_negative").value}); show("AI 与提示词设置已保存"); } catch (e) { show(e.message); }
};
document.getElementById("draw_reply_mode").onchange = updateReplyCustomVisibility;
document.getElementById("saveReply").onclick = async () => { try { await post(`${API}/config`, {draw_reply_mode: document.getElementById("draw_reply_mode").value, draw_delivery_mode: document.getElementById("draw_delivery_mode").value, draw_reply_timeout: Number(document.getElementById("draw_reply_timeout").value || 6), draw_start_reply: document.getElementById("draw_start_reply").value, draw_reply_custom: document.getElementById("draw_reply_custom").value}); show("回复设置已保存"); } catch (e) { show(e.message); } };
document.getElementById("addPreset").onclick = async () => {
  const name = document.getElementById("presetName").value;
  const content = document.getElementById("presetContent").value;
  const wasEditing = !!editingPreset;
  try { await post(`${API}/presets`, {action: wasEditing ? "update" : "add", name, content}); resetPresetEditor(); await loadPresets(); show(wasEditing ? "预设已修改" : "预设已添加"); } catch (e) { show(e.message); }
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
