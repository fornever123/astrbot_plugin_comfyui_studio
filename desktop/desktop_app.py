from __future__ import annotations

import asyncio
import json
import os
import queue
import random
import threading
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 允许直接运行 desktop/desktop_app.py，而不要求用户先安装成 Python 包。
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PLUGIN_ROOT))

from ai import AIError, AITranslator  # noqa: E402
from anima_knowledge import build_context  # noqa: E402
from comfy import ComfyClient, ComfyError  # noqa: E402
from workflow import (  # noqa: E402
    WorkflowError,
    adapt_generic,
    infer_workflow_mapping,
    inspect_workflow,
    load_workflow,
    validate_workflow_mapping,
)


APP_NAME = "Anima ComfyUI 桌面端"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW_DIR = PLUGIN_ROOT / "workflows"
DEFAULT_OUTPUT_DIR = PLUGIN_ROOT / "output" / "desktop"

CHAT_SYSTEM_PROMPT = """你是 Anima ComfyUI 桌面端的中文绘图助手。
你可以正常聊天，也可以把用户的自然语言请求转换成绘图任务。
只输出一个 JSON 对象，不要 Markdown，不要解释 JSON 以外的内容。
普通聊天格式：{"action":"chat","reply":"中文回复"}
绘图格式：{"action":"draw","mode":"txt2img 或 img2img","prompt":"实际发送给工作流的提示词","negative_prompt":"负面提示词","parameters":{"width":1024,"height":1024,"steps":20,"cfg":5,"seed":-1,"sampler_name":"euler","scheduler":"normal","denoise":1}}
用户明确要求修改已有图片时使用 img2img；没有参考图时使用 txt2img。
绘图 prompt 必须保留用户明确提出的主体、动作、姿势、服装、表情、镜头和场景，不要把“换个姿势”偷换成没有动作的空句。
所有回复内容使用中文；prompt 可以根据工作流需要使用英文标签，但不要输出思考过程。"""

WORKFLOW_AI_SYSTEM_PROMPT = """你是 ComfyUI 工作流适配工程师。
根据用户提供的节点摘要，返回一个 JSON 映射，只能引用摘要中存在的节点 ID 和输入名。
不要修改节点结构，不要输出解释，不要输出 Markdown。
JSON 字段必须是：
positive_text、negative_text、images、model、clip、vae、loras、samplers、sizes、outputs。
文本/图片/模型/编码器/VAE/LoRA 数组元素格式为 {"node_id":"节点ID","input":"输入名"}。
samplers 和 sizes 元素格式为 {"node_id":"节点ID","inputs":["字段名"]}。
outputs 是输出节点 ID 字符串数组。
positive_text 选择正面提示词，negative_text 选择负面提示词，images 选择 LoadImage 等参考图输入。
如果某个类别没有候选，返回空数组。"""

ANIMA_CONTEXT = build_context(PLUGIN_ROOT, maximum=2600)


def _app_config_path() -> Path:
    appdata = str(os.environ.get("APPDATA", "")).strip()
    root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return root / "AnimaComfyUIStudio" / "desktop.json"


class ConfigStore:
    def __init__(self) -> None:
        self.path = _app_config_path()
        self.data: dict[str, Any] = {
            "comfy_url": DEFAULT_COMFY_URL,
            "ai_base_url": "",
            "ai_api_key": "",
            "ai_model": "",
            "workflow_dir": str(DEFAULT_WORKFLOW_DIR),
            "output_dir": str(DEFAULT_OUTPUT_DIR),
            "txt2img_workflow": "",
            "img2img_workflow": "",
            "workflow_mappings": {},
            "chat_history": [],
        }
        self.load()

    def load(self) -> None:
        try:
            if self.path.is_file():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.data.update(raw)
        except (OSError, ValueError):
            # 配置损坏时保留默认值，桌面端仍应能启动并让用户重新保存。
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_values(value: str) -> list[str]:
    return [item.strip() for item in value.replace("，", ",").replace("\n", ",").split(",") if item.strip()]


def _int_value(value: str, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _float_value(value: str, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _extract_json(value: str) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("``")).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except ValueError:
            return None
    return None


def _normalize_ai_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    result = dict(mapping)
    outputs = result.get("outputs", [])
    if isinstance(outputs, list):
        result["outputs"] = [
            item.get("node_id") if isinstance(item, dict) else item
            for item in outputs
        ]
    return result


class DesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1360x900")
        self.minsize(1080, 720)
        self.store = ConfigStore()
        self._tasks: queue.Queue[tuple[str, Any, Exception | None]] = queue.Queue()
        self._busy = 0
        self._workflow_paths: dict[str, Path] = {}
        self._workflow_display: dict[str, str] = {}
        self._last_workflow_json = ""
        self._build_style()
        self._build_vars()
        self._build_ui()
        self.refresh_workflows()
        self.after(100, self._drain_tasks)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#687386")
        style.configure("Accent.TButton", padding=(14, 7))
        style.configure("Card.TLabelframe", padding=10)

    def _build_vars(self) -> None:
        data = self.store.data
        self.comfy_url = tk.StringVar(value=str(data.get("comfy_url", DEFAULT_COMFY_URL)))
        self.ai_base_url = tk.StringVar(value=str(data.get("ai_base_url", "")))
        self.ai_api_key = tk.StringVar(value=str(data.get("ai_api_key", "")))
        self.ai_model = tk.StringVar(value=str(data.get("ai_model", "")))
        self.workflow_dir = tk.StringVar(value=str(data.get("workflow_dir", DEFAULT_WORKFLOW_DIR)))
        self.output_dir = tk.StringVar(value=str(data.get("output_dir", DEFAULT_OUTPUT_DIR)))
        self.txt_workflow = tk.StringVar(value=str(data.get("txt2img_workflow", "")))
        self.img_workflow = tk.StringVar(value=str(data.get("img2img_workflow", "")))
        self.txt_prompt = tk.StringVar()
        self.txt_negative = tk.StringVar()
        self.img_prompt = tk.StringVar()
        self.img_negative = tk.StringVar()
        self.txt_model = tk.StringVar()
        self.img_model = tk.StringVar()
        self.txt_clip = tk.StringVar()
        self.img_clip = tk.StringVar()
        self.txt_vae = tk.StringVar()
        self.img_vae = tk.StringVar()
        self.txt_loras = tk.StringVar()
        self.img_loras = tk.StringVar()
        self.draw_params: dict[str, dict[str, tk.StringVar]] = {}
        for mode in ("txt2img", "img2img"):
            self.draw_params[mode] = {
                "width": tk.StringVar(value="1024"),
                "height": tk.StringVar(value="1024"),
                "steps": tk.StringVar(value="20"),
                "cfg": tk.StringVar(value="5"),
                "seed": tk.StringVar(value="-1"),
                "sampler": tk.StringVar(value="euler"),
                "scheduler": tk.StringVar(value="normal"),
                "denoise": tk.StringVar(value="1.0"),
                "batch": tk.StringVar(value="1"),
            }
        self.image_paths = tk.StringVar()
        self.workflow_mode = tk.StringVar(value="文生图")
        self.status_text = tk.StringVar(value="就绪")

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(18, 14, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.status_text, style="Hint.TLabel").pack(side="right")

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.chat_tab = ttk.Frame(self.tabs, padding=12)
        self.txt_tab = ttk.Frame(self.tabs, padding=12)
        self.img_tab = ttk.Frame(self.tabs, padding=12)
        self.workflow_tab = ttk.Frame(self.tabs, padding=12)
        self.comfy_tab = ttk.Frame(self.tabs, padding=12)
        self.settings_tab = ttk.Frame(self.tabs, padding=12)
        for title, frame in (
            ("对话", self.chat_tab),
            ("文生图", self.txt_tab),
            ("图生图", self.img_tab),
            ("工作流", self.workflow_tab),
            ("ComfyUI", self.comfy_tab),
            ("设置", self.settings_tab),
        ):
            self.tabs.add(frame, text=title)
        self._build_chat_tab()
        self._build_draw_tab(self.txt_tab, "txt2img")
        self._build_draw_tab(self.img_tab, "img2img")
        self._build_workflow_tab()
        self._build_comfy_tab()
        self._build_settings_tab()

    def _text_box(self, parent: tk.Misc, height: int = 12) -> tk.Text:
        box = tk.Text(parent, height=height, wrap="word", undo=True, font=("Consolas", 10))
        box.configure(relief="solid", borderwidth=1, padx=8, pady=8)
        return box

    def _build_chat_tab(self) -> None:
        self.chat_log = self._text_box(self.chat_tab, 28)
        self.chat_log.pack(fill="both", expand=True)
        self.chat_log.configure(state="disabled")
        bottom = ttk.Frame(self.chat_tab)
        bottom.pack(fill="x", pady=(10, 0))
        self.chat_input = ttk.Entry(bottom)
        self.chat_input.pack(side="left", fill="x", expand=True)
        self.chat_input.bind("<Return>", lambda _event: self.send_chat())
        ttk.Button(bottom, text="发送", style="Accent.TButton", command=self.send_chat).pack(side="left", padx=(8, 0))
        ttk.Label(
            self.chat_tab,
            text="对话 AI 可以直接决定聊天或调用当前工作流绘图；AI 配置在“设置”中填写。",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(8, 0))
        for item in self.store.data.get("chat_history", [])[-20:]:
            if isinstance(item, dict):
                self._append_chat(str(item.get("role", "")), str(item.get("content", "")))

    def _build_draw_tab(self, parent: ttk.Frame, mode: str) -> None:
        top = ttk.LabelFrame(parent, text="工作流与提示词", style="Card.TLabelframe")
        top.pack(fill="x")
        row = 0
        ttk.Label(top, text="调用工作流").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        combo = ttk.Combobox(top, state="readonly", width=75)
        combo.grid(row=row, column=1, columnspan=4, sticky="ew", padx=4, pady=4)
        setattr(self, f"{mode}_workflow_combo", combo)
        ttk.Button(top, text="刷新", command=self.refresh_workflows).grid(row=row, column=5, padx=4)
        row += 1
        ttk.Label(top, text="正面提示词").grid(row=row, column=0, sticky="nw", padx=4, pady=4)
        prompt = ttk.Entry(top)
        prompt.grid(row=row, column=1, columnspan=5, sticky="ew", padx=4, pady=4)
        prompt_var = self.txt_prompt if mode == "txt2img" else self.img_prompt
        prompt.configure(textvariable=prompt_var)
        row += 1
        ttk.Label(top, text="负面提示词").grid(row=row, column=0, sticky="nw", padx=4, pady=4)
        negative = ttk.Entry(top)
        negative.grid(row=row, column=1, columnspan=5, sticky="ew", padx=4, pady=4)
        negative.configure(textvariable=self.txt_negative if mode == "txt2img" else self.img_negative)
        for column in range(1, 6):
            top.columnconfigure(column, weight=1)

        if mode == "img2img":
            row += 1
            ttk.Label(top, text="参考图片").grid(row=row, column=0, sticky="w", padx=4, pady=4)
            ttk.Entry(top, textvariable=self.image_paths).grid(row=row, column=1, columnspan=4, sticky="ew", padx=4, pady=4)
            ttk.Button(top, text="选择图片（最多 3 张）", command=self.choose_images).grid(row=row, column=5, padx=4)

        model_var = self.txt_model if mode == "txt2img" else self.img_model
        clip_var = self.txt_clip if mode == "txt2img" else self.img_clip
        vae_var = self.txt_vae if mode == "txt2img" else self.img_vae
        lora_var = self.txt_loras if mode == "txt2img" else self.img_loras
        row += 1
        for label, variable in (("核心模型", model_var), ("文本编码器", clip_var), ("VAE", vae_var), ("LoRA（逗号分隔）", lora_var)):
            ttk.Label(top, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
            ttk.Entry(top, textvariable=variable).grid(row=row, column=1, columnspan=5, sticky="ew", padx=4, pady=4)
            row += 1

        params = ttk.LabelFrame(parent, text="采样参数", style="Card.TLabelframe")
        params.pack(fill="x", pady=(12, 0))
        mode_params = self.draw_params[mode]
        fields = (
            ("宽度", mode_params["width"]), ("高度", mode_params["height"]), ("步数", mode_params["steps"]),
            ("CFG", mode_params["cfg"]), ("种子（-1 随机）", mode_params["seed"]), ("采样器", mode_params["sampler"]),
            ("调度器", mode_params["scheduler"]), ("重绘幅度", mode_params["denoise"]), ("批量数", mode_params["batch"]),
        )
        for index, (label, variable) in enumerate(fields):
            row, column = divmod(index, 6)
            ttk.Label(params, text=label).grid(row=row * 2, column=column, sticky="w", padx=4, pady=(5, 0))
            ttk.Entry(params, textvariable=variable, width=16).grid(row=row * 2 + 1, column=column, sticky="ew", padx=4, pady=(0, 5))
            params.columnconfigure(column, weight=1)

        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(12, 0))
        ttk.Button(controls, text="开始绘图", style="Accent.TButton", command=lambda: self.start_draw(mode)).pack(side="left")
        ttk.Button(controls, text="查看当前工作流", command=self.show_last_workflow).pack(side="left", padx=8)
        ttk.Label(
            parent,
            text=("图生图会把选择的图片上传到 ComfyUI，并由工作流映射决定实际输入节点；"
                  "未适配的工作流请先进入“工作流”页执行 AI 一键适配。" if mode == "img2img" else
                  "文生图会保留工作流原有节点，只覆盖识别到的提示词、模型和采样参数。"),
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(8, 0))

    def _build_workflow_tab(self) -> None:
        controls = ttk.Frame(self.workflow_tab)
        controls.pack(fill="x")
        ttk.Label(controls, text="工作流").pack(side="left")
        self.workflow_combo = ttk.Combobox(controls, state="readonly", width=70)
        self.workflow_combo.pack(side="left", fill="x", expand=True, padx=8)
        self.workflow_combo.bind("<<ComboboxSelected>>", lambda _event: self.inspect_selected_workflow())
        ttk.Button(controls, text="刷新", command=self.refresh_workflows).pack(side="left")
        ttk.Button(controls, text="查看节点", command=self.inspect_selected_workflow).pack(side="left", padx=6)
        ttk.Button(controls, text="AI 一键适配", style="Accent.TButton", command=lambda: self.adapt_selected_workflow(True)).pack(side="left")
        ttk.Button(controls, text="离线自动适配", command=lambda: self.adapt_selected_workflow(False)).pack(side="left", padx=6)

        assignment = ttk.Frame(self.workflow_tab)
        assignment.pack(fill="x", pady=(10, 0))
        ttk.Label(assignment, text="适用模式").pack(side="left")
        ttk.Combobox(assignment, textvariable=self.workflow_mode, values=["文生图", "图生图"], state="readonly", width=12).pack(side="left", padx=8)
        ttk.Button(assignment, text="设为当前模式工作流", command=self.assign_selected_workflow).pack(side="left")
        ttk.Button(assignment, text="保存映射 JSON", command=self.save_mapping_text).pack(side="left", padx=8)

        panes = ttk.Panedwindow(self.workflow_tab, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.LabelFrame(panes, text="节点摘要", style="Card.TLabelframe")
        right = ttk.LabelFrame(panes, text="当前映射（可编辑）", style="Card.TLabelframe")
        panes.add(left, weight=3)
        panes.add(right, weight=2)
        self.workflow_summary = self._text_box(left, 30)
        self.workflow_summary.pack(fill="both", expand=True)
        self.workflow_mapping = self._text_box(right, 30)
        self.workflow_mapping.pack(fill="both", expand=True)

    def _build_comfy_tab(self) -> None:
        controls = ttk.Frame(self.comfy_tab)
        controls.pack(fill="x")
        ttk.Button(controls, text="检查 ComfyUI", style="Accent.TButton", command=self.check_comfy).pack(side="left")
        ttk.Button(controls, text="查看队列", command=self.check_queue).pack(side="left", padx=8)
        ttk.Button(controls, text="打开 localhost:8188", command=lambda: webbrowser.open(self.comfy_url.get().strip() or DEFAULT_COMFY_URL)).pack(side="left")
        ttk.Button(controls, text="查看当前调用工作流", command=self.show_last_workflow).pack(side="left", padx=8)
        self.comfy_log = self._text_box(self.comfy_tab, 30)
        self.comfy_log.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(self.comfy_tab, text="这里显示桌面端最后提交的工作流和 ComfyUI 状态；打开按钮会在浏览器中访问本地 8188 页面。", style="Hint.TLabel").pack(anchor="w", pady=(8, 0))

    def _build_settings_tab(self) -> None:
        frame = ttk.LabelFrame(self.settings_tab, text="连接设置", style="Card.TLabelframe")
        frame.pack(fill="x")
        entries = (
            ("ComfyUI 服务地址", self.comfy_url, False),
            ("AI 服务地址（OpenAI 兼容）", self.ai_base_url, False),
            ("AI API Key", self.ai_api_key, True),
            ("AI 模型", self.ai_model, False),
            ("工作流目录", self.workflow_dir, False),
            ("桌面端输出目录", self.output_dir, False),
        )
        for row, (label, variable, secret) in enumerate(entries):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=6)
            ttk.Entry(frame, textvariable=variable, show="*" if secret else "").grid(row=row, column=1, sticky="ew", padx=5, pady=6)
            if label in {"工作流目录", "桌面端输出目录"}:
                ttk.Button(frame, text="浏览", command=lambda var=variable: self.browse_directory(var)).grid(row=row, column=2, padx=5)
        frame.columnconfigure(1, weight=1)
        buttons = ttk.Frame(self.settings_tab)
        buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(buttons, text="保存设置", style="Accent.TButton", command=self.save_settings).pack(side="left")
        ttk.Button(buttons, text="获取 AI 模型列表", command=self.list_ai_models).pack(side="left", padx=8)
        ttk.Button(buttons, text="测试 AI 连接", command=self.test_ai).pack(side="left")
        self.ai_models_combo = ttk.Combobox(self.settings_tab, textvariable=self.ai_model, state="normal")
        self.ai_models_combo.pack(fill="x", pady=(12, 0))
        ttk.Label(self.settings_tab, text="API Key 只保存在本机桌面端配置文件，不会写入工作流 JSON。", style="Hint.TLabel").pack(anchor="w", pady=(8, 0))

    def _append_chat(self, role: str, content: str) -> None:
        self.chat_log.configure(state="normal")
        label = "你" if role == "user" else "AI"
        self.chat_log.insert("end", f"{label}：{content}\n\n")
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)

    def _selected_display(self, mode: str) -> str:
        combo: ttk.Combobox = getattr(self, f"{mode}_workflow_combo")
        return str(combo.get() or "")

    def _selected_path(self, display: str | None = None) -> Path:
        value = display or self.workflow_combo.get()
        path = self._workflow_paths.get(str(value))
        if path is None:
            raise WorkflowError("请先选择一个工作流")
        return path

    def refresh_workflows(self) -> None:
        root = Path(self.workflow_dir.get().strip() or str(DEFAULT_WORKFLOW_DIR))
        paths: list[Path] = []
        if root.is_dir():
            paths.extend(sorted(root.glob("*.json")))
        if root.resolve() != DEFAULT_WORKFLOW_DIR.resolve() and DEFAULT_WORKFLOW_DIR.is_dir():
            paths.extend(sorted(DEFAULT_WORKFLOW_DIR.glob("*.json")))
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve()).casefold()
            if key not in seen:
                seen.add(key)
                unique.append(path)
        self._workflow_paths.clear()
        self._workflow_display.clear()
        displays = []
        for path in unique:
            display = f"{path.name}  |  {path}"
            self._workflow_paths[display] = path
            self._workflow_display[str(path.resolve())] = display
            displays.append(display)
        for mode in ("txt2img", "img2img"):
            combo: ttk.Combobox = getattr(self, f"{mode}_workflow_combo")
            combo["values"] = displays
            configured = str(self.txt_workflow.get() if mode == "txt2img" else self.img_workflow.get())
            selected = self._workflow_display.get(str(Path(configured).resolve())) if configured else ""
            if selected in displays:
                combo.set(selected)
            elif displays:
                combo.set(displays[0])
        self.workflow_combo["values"] = displays
        if self.workflow_combo.get() not in displays and displays:
            self.workflow_combo.set(displays[0])
        self.status_text.set(f"已发现 {len(displays)} 个工作流")

    def browse_directory(self, variable: tk.StringVar) -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            variable.set(chosen)
            if variable is self.workflow_dir:
                self.refresh_workflows()

    def choose_images(self) -> None:
        selected = filedialog.askopenfilenames(
            title="选择参考图片（最多 3 张）",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp"), ("所有文件", "*.*")],
        )
        if selected:
            self.image_paths.set("\n".join(selected[:3]))

    def save_settings(self) -> None:
        self.store.data.update({
            "comfy_url": self.comfy_url.get().strip() or DEFAULT_COMFY_URL,
            "ai_base_url": self.ai_base_url.get().strip(),
            "ai_api_key": self.ai_api_key.get(),
            "ai_model": self.ai_model.get().strip(),
            "workflow_dir": self.workflow_dir.get().strip() or str(DEFAULT_WORKFLOW_DIR),
            "output_dir": self.output_dir.get().strip() or str(DEFAULT_OUTPUT_DIR),
            "txt2img_workflow": self.txt_workflow.get().strip(),
            "img2img_workflow": self.img_workflow.get().strip(),
        })
        try:
            self.store.save()
            self.refresh_workflows()
            self.status_text.set(f"设置已保存：{self.store.path}")
        except OSError as exc:
            messagebox.showerror("保存失败", f"无法保存桌面端设置：{exc}")

    def _start_task(self, label: str, func: Callable[[], Any]) -> None:
        self._busy += 1
        self.status_text.set(f"正在{label}…")

        def runner() -> None:
            try:
                value = func()
                self._tasks.put((label, value, None))
            except Exception as exc:  # noqa: BLE001
                self._tasks.put((label, None, exc))

        threading.Thread(target=runner, daemon=True).start()

    def _drain_tasks(self) -> None:
        while True:
            try:
                label, value, error = self._tasks.get_nowait()
            except queue.Empty:
                break
            self._busy = max(0, self._busy - 1)
            if error is not None:
                self.status_text.set(f"{label}失败")
                self._show_error(label, error)
                continue
            self.status_text.set(f"{label}完成")
            self._handle_task_result(label, value)
        self.after(100, self._drain_tasks)

    def _show_error(self, label: str, error: Exception) -> None:
        message = str(error) or type(error).__name__
        if self._busy:
            self._append_chat("assistant", f"{label}失败：{message}")
        else:
            messagebox.showerror(f"{label}失败", message)
        self._set_text(self.comfy_log, message)

    def _handle_task_result(self, label: str, value: Any) -> None:
        if label in {"文生图", "图生图"} and isinstance(value, dict):
            paths = value.get("images", [])
            report = "；".join(value.get("report", []))
            message = f"{label}完成，共 {len(paths)} 张图片。\n" + "\n".join(str(path) for path in paths)
            if report:
                message += f"\n适配报告：{report}"
            self._append_chat("assistant", message)
            self._set_text(self.comfy_log, self._last_workflow_json)
        elif label == "AI 对话" and isinstance(value, dict):
            self._finish_chat(value)
        elif label == "AI 一键适配" and isinstance(value, dict):
            self._set_text(self.workflow_summary, json.dumps(value, ensure_ascii=False, indent=2))
            self._set_text(self.workflow_mapping, json.dumps(value.get("mapping", {}), ensure_ascii=False, indent=2))
        elif label == "离线自动适配" and isinstance(value, dict):
            self._set_text(self.workflow_summary, json.dumps(value, ensure_ascii=False, indent=2))
            self._set_text(self.workflow_mapping, json.dumps(value.get("mapping", {}), ensure_ascii=False, indent=2))
        elif label == "ComfyUI 状态":
            self._set_text(self.comfy_log, json.dumps(value, ensure_ascii=False, indent=2))
        elif label == "ComfyUI 队列":
            self._set_text(self.comfy_log, json.dumps(value, ensure_ascii=False, indent=2))
        elif label == "AI 模型列表":
            self.ai_models_combo["values"] = value or []
            if value and not self.ai_model.get():
                self.ai_model.set(value[0])
        elif label == "AI 连接测试":
            messagebox.showinfo("AI 连接测试", str(value))

    def _ai_config(self) -> dict[str, Any]:
        return {
            "ai_base_url": self.ai_base_url.get().strip(),
            "ai_api_key": self.ai_api_key.get(),
            "ai_model": self.ai_model.get().strip(),
            "ai_enabled": True,
        }

    def _request_ai(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 1200,
        config: dict[str, Any] | None = None,
    ) -> str:
        translator = AITranslator(config or self._ai_config())
        return asyncio.run(translator.generate(
            user_prompt,
            system_prompt=system_prompt,
            require_enabled=True,
            max_tokens=max_tokens,
            preserve_newlines=True,
        ))

    def send_chat(self) -> None:
        content = self.chat_input.get().strip()
        if not content:
            return
        self.chat_input.delete(0, "end")
        self._append_chat("user", content)
        history = self.store.data.setdefault("chat_history", [])
        if isinstance(history, list):
            history.extend([{"role": "user", "content": content}])
            del history[:-20]
        self._start_task("AI 对话", lambda: self._chat_task(content, self._draw_state()))

    def _chat_task(self, content: str, draw_state: dict[str, Any]) -> dict[str, Any]:
        history = self.store.data.get("chat_history", [])
        context = "\n".join(
            f"{item.get('role')}: {item.get('content')}"
            for item in history[-8:]
            if isinstance(item, dict)
        )
        system_prompt = (
            CHAT_SYSTEM_PROMPT
            + "\n\n当 action=draw 时，prompt 字段要遵循下面的 Anima 提示词工程规则；"
              "只把规则用于 prompt 字段，不要把规则原文放进 reply 或 JSON 外内容。\n"
            + ANIMA_CONTEXT
        )
        raw = self._request_ai(
            f"最近对话：\n{context}\n\n本次用户消息：\n{content}",
            system_prompt,
            config=draw_state["ai_config"],
        )
        result = _extract_json(raw)
        if result is None:
            return {"action": "chat", "reply": raw.strip()}
        if str(result.get("action", "chat")).lower() != "draw":
            return {"action": "chat", "reply": str(result.get("reply", "我暂时没有理解这句话。"))}
        mode = "img2img" if str(result.get("mode", "txt2img")).lower() == "img2img" else "txt2img"
        parameters = result.get("parameters") if isinstance(result.get("parameters"), dict) else {}
        return {
            "action": "draw",
            "reply": str(result.get("reply", "我来开始绘图。")),
            "draw": self._execute_draw(
                mode,
                str(result.get("prompt", "")).strip(),
                str(result.get("negative_prompt", "")).strip(),
                parameters,
                draw_state["image_paths"] if mode == "img2img" else [],
                state=draw_state,
            ),
        }

    def _finish_chat(self, result: dict[str, Any]) -> None:
        reply = str(result.get("reply", ""))
        if reply:
            self._append_chat("assistant", reply)
            history = self.store.data.setdefault("chat_history", [])
            if isinstance(history, list):
                history.append({"role": "assistant", "content": reply})
                del history[:-20]
        if result.get("action") == "draw":
            draw = result.get("draw")
            if isinstance(draw, dict):
                paths = draw.get("images", [])
                self._append_chat("assistant", "绘图完成：\n" + "\n".join(str(path) for path in paths))
        try:
            self.store.save()
        except OSError:
            pass

    def _current_images(self) -> list[str]:
        return [item for item in self.image_paths.get().splitlines() if item.strip()][:3]

    def _workflow_for_mode(self, mode: str) -> Path:
        display = self._selected_display(mode)
        return self._selected_path(display)

    def _draw_state(self) -> dict[str, Any]:
        """在主线程读取 Tk 控件，后台任务只使用这个普通字典。"""
        workflow_paths: dict[str, str] = {}
        for mode in ("txt2img", "img2img"):
            display = self._selected_display(mode)
            path = self._workflow_paths.get(display)
            workflow_paths[mode] = str(path) if path else ""
        parameters: dict[str, dict[str, str]] = {}
        for mode, values in self.draw_params.items():
            parameters[mode] = {key: variable.get() for key, variable in values.items()}
        return {
            "comfy_url": self.comfy_url.get().strip() or DEFAULT_COMFY_URL,
            "output_dir": self.output_dir.get().strip() or str(DEFAULT_OUTPUT_DIR),
            "workflow_paths": workflow_paths,
            "models": {"txt2img": self.txt_model.get(), "img2img": self.img_model.get()},
            "clips": {"txt2img": self.txt_clip.get(), "img2img": self.img_clip.get()},
            "vaes": {"txt2img": self.txt_vae.get(), "img2img": self.img_vae.get()},
            "loras": {"txt2img": self.txt_loras.get(), "img2img": self.img_loras.get()},
            "parameters": parameters,
            "image_paths": self._current_images(),
            "ai_config": self._ai_config(),
        }

    def _mapping_for(self, path: Path, mode: str) -> dict[str, Any]:
        mappings = self.store.data.get("workflow_mappings", {})
        key = str(path.resolve())
        mapping = mappings.get(key) if isinstance(mappings, dict) else None
        if not isinstance(mapping, dict):
            mapping = infer_workflow_mapping(load_workflow(path), mode)
        return mapping

    def _execute_draw(
        self,
        mode: str,
        positive: str,
        negative: str,
        parameters: dict[str, Any],
        image_paths: list[str],
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not positive:
            raise WorkflowError("请提供画面描述或提示词")
        state = state or self._draw_state()
        raw_path = str(state.get("workflow_paths", {}).get(mode, ""))
        if not raw_path:
            raise WorkflowError(f"请先在工作流页为{('文生图' if mode == 'txt2img' else '图生图')}选择工作流")
        path = Path(raw_path)
        source = load_workflow(path)
        mapping = self._mapping_for(path, mode)
        uploaded: list[str] = []
        client = ComfyClient(str(state.get("comfy_url") or DEFAULT_COMFY_URL), timeout=120)
        defaults = state.get("parameters", {}).get(mode, {})
        if not isinstance(defaults, dict):
            defaults = {}

        async def upload_all() -> None:
            for image_path in image_paths[:3]:
                info = await client.upload_image(image_path)
                name = str(info.get("name", ""))
                subfolder = str(info.get("subfolder", "")).strip("/")
                uploaded.append(f"{subfolder}/{name}" if subfolder else name)

        if mode == "img2img":
            if not image_paths:
                raise WorkflowError("图生图至少需要选择一张参考图片")
            asyncio.run(upload_all())
        seed = _int_value(str(parameters.get("seed", defaults.get("seed", "-1"))), -1)
        graph, report = adapt_generic(
            source,
            mode=mode,
            positive=positive,
            negative=negative,
            image_names=uploaded,
            model_name=str(parameters.get("model", state["models"][mode]) or ""),
            clip_name=str(parameters.get("clip", state["clips"][mode]) or ""),
            vae_name=str(parameters.get("vae", state["vaes"][mode]) or ""),
            loras=_split_values(str(parameters.get("loras", state["loras"][mode]) or "")),
            width=_int_value(str(parameters.get("width", defaults.get("width", "1024"))), 1024),
            height=_int_value(str(parameters.get("height", defaults.get("height", "1024"))), 1024),
            steps=_int_value(str(parameters.get("steps", defaults.get("steps", "20"))), 20),
            cfg=_float_value(str(parameters.get("cfg", defaults.get("cfg", "5"))), 5.0),
            seed=seed,
            sampler_name=str(parameters.get("sampler_name", defaults.get("sampler", "euler")) or "euler"),
            scheduler=str(parameters.get("scheduler", defaults.get("scheduler", "normal")) or "normal"),
            denoise=_float_value(str(parameters.get("denoise", defaults.get("denoise", "1.0"))), 1.0),
            batch=_int_value(str(parameters.get("batch", defaults.get("batch", "1"))), 1),
            filename_prefix=f"astrbot/desktop_{mode}",
            mapping=mapping,
        )
        self._last_workflow_json = json.dumps(graph, ensure_ascii=False, indent=2)

        async def run() -> list[Path]:
            prompt_id = await client.queue(graph)
            history = await client.wait(prompt_id, timeout=900)
            return await client.download_outputs(history, Path(str(state.get("output_dir") or DEFAULT_OUTPUT_DIR)))

        images = asyncio.run(run())
        if not images:
            raise ComfyError("ComfyUI 已完成任务，但没有找到 SaveImage 输出")
        return {"images": [str(item) for item in images], "report": report}

    def _model_for_mode(self, mode: str) -> str:
        return self.txt_model.get() if mode == "txt2img" else self.img_model.get()

    def _clip_for_mode(self, mode: str) -> str:
        return self.txt_clip.get() if mode == "txt2img" else self.img_clip.get()

    def _vae_for_mode(self, mode: str) -> str:
        return self.txt_vae.get() if mode == "txt2img" else self.img_vae.get()

    def _loras_for_mode(self, mode: str) -> str:
        return self.txt_loras.get() if mode == "txt2img" else self.img_loras.get()

    def start_draw(self, mode: str) -> None:
        positive = self.txt_prompt.get().strip() if mode == "txt2img" else self.img_prompt.get().strip()
        negative = self.txt_negative.get().strip() if mode == "txt2img" else self.img_negative.get().strip()
        images = self._current_images() if mode == "img2img" else []
        state = self._draw_state()
        self._start_task("文生图" if mode == "txt2img" else "图生图", lambda: self._execute_draw(mode, positive, negative, {}, images, state=state))

    def inspect_selected_workflow(self) -> None:
        try:
            path = self._selected_path()
            summary = inspect_workflow(load_workflow(path))
            self._set_text(self.workflow_summary, json.dumps(summary, ensure_ascii=False, indent=2))
            mapping = self._mapping_for(path, self.workflow_mode.get() == "图生图" and "img2img" or "txt2img")
            self._set_text(self.workflow_mapping, json.dumps(mapping, ensure_ascii=False, indent=2))
        except Exception as exc:  # noqa: BLE001
            self._show_error("读取工作流", exc)

    def adapt_selected_workflow(self, use_ai: bool) -> None:
        try:
            path = self._selected_path()
        except Exception as exc:  # noqa: BLE001
            self._show_error("适配工作流", exc)
            return
        mode = "img2img" if self.workflow_mode.get() == "图生图" else "txt2img"
        label = "AI 一键适配" if use_ai else "离线自动适配"
        ai_config = self._ai_config()
        self._start_task(label, lambda: self._adapt_task(path, mode, use_ai, ai_config))

    def _adapt_task(
        self,
        path: Path,
        mode: str,
        use_ai: bool,
        ai_config: dict[str, Any],
    ) -> dict[str, Any]:
        source = load_workflow(path)
        summary = inspect_workflow(source)
        mapping: dict[str, Any]
        ai_error = ""
        if use_ai:
            try:
                raw = self._request_ai(
                    "请分析下面的 ComfyUI 工作流节点摘要并返回映射 JSON：\n" + json.dumps(summary, ensure_ascii=False)[:28000],
                    WORKFLOW_AI_SYSTEM_PROMPT,
                    max_tokens=1800,
                    config=ai_config,
                )
                parsed = _extract_json(raw)
                if parsed is None:
                    raise AIError("AI 返回内容不是有效 JSON")
                mapping = _normalize_ai_mapping(parsed)
            except Exception as exc:  # noqa: BLE001
                ai_error = f"AI 适配不可用，已使用离线识别：{exc}"
                mapping = infer_workflow_mapping(source, mode)
        else:
            mapping = infer_workflow_mapping(source, mode)
        validated, warnings = validate_workflow_mapping(source, mapping)
        if not validated.get("positive_text"):
            fallback = infer_workflow_mapping(source, mode)
            for key, value in fallback.items():
                if not validated.get(key):
                    validated[key] = value
        key = str(path.resolve())
        mappings = self.store.data.setdefault("workflow_mappings", {})
        if not isinstance(mappings, dict):
            mappings = {}
            self.store.data["workflow_mappings"] = mappings
        mappings[key] = validated
        self.store.save()
        if ai_error:
            warnings.insert(0, ai_error)
        validated["warnings"] = warnings
        return {"summary": summary, "mapping": validated}

    def save_mapping_text(self) -> None:
        try:
            path = self._selected_path()
            raw = json.loads(self.workflow_mapping.get("1.0", "end"))
            if not isinstance(raw, dict):
                raise ValueError("映射必须是 JSON 对象")
            mapping, warnings = validate_workflow_mapping(load_workflow(path), raw)
            if not mapping.get("positive_text") or not mapping.get("outputs"):
                raise WorkflowError("映射至少需要有效的正面提示词和输出节点")
            self.store.data.setdefault("workflow_mappings", {})[str(path.resolve())] = mapping
            self.store.save()
            self._set_text(self.workflow_mapping, json.dumps(mapping, ensure_ascii=False, indent=2))
            self.status_text.set("工作流映射已保存" + (f"；{';'.join(warnings)}" if warnings else ""))
        except (ValueError, OSError, WorkflowError) as exc:
            self._show_error("保存映射", exc)

    def assign_selected_workflow(self) -> None:
        try:
            path = self._selected_path()
            if self.workflow_mode.get() == "图生图":
                self.img_workflow.set(str(path))
            else:
                self.txt_workflow.set(str(path))
            self.save_settings()
        except Exception as exc:  # noqa: BLE001
            self._show_error("设置当前工作流", exc)

    def show_last_workflow(self) -> None:
        if not self._last_workflow_json:
            self._set_text(self.comfy_log, "本次桌面端尚未提交工作流。")
            self.tabs.select(self.comfy_tab)
            return
        self._set_text(self.comfy_log, self._last_workflow_json)
        self.tabs.select(self.comfy_tab)

    def check_comfy(self) -> None:
        url = self.comfy_url.get().strip() or DEFAULT_COMFY_URL
        self._start_task("ComfyUI 状态", lambda: asyncio.run(ComfyClient(url).status()))

    def check_queue(self) -> None:
        url = self.comfy_url.get().strip() or DEFAULT_COMFY_URL
        self._start_task("ComfyUI 队列", lambda: asyncio.run(ComfyClient(url).queue_info()))

    def list_ai_models(self) -> None:
        config = self._ai_config()
        self._start_task("AI 模型列表", lambda: asyncio.run(AITranslator(config).list_models()))

    def test_ai(self) -> None:
        config = self._ai_config()

        def task() -> str:
            asyncio.run(AITranslator(config).test_connection())
            return "连接成功，模型响应正常。"

        self._start_task("AI 连接测试", task)


def main() -> None:
    try:
        app = DesktopApp()
        app.mainloop()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
