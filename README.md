# ComfyUI AI 绘画台

[![版本](https://img.shields.io/badge/版本-0.6.8-2f80ed.svg)](https://github.com/fornever123/astrbot_plugin_comfyui_studio)
[![AstrBot](https://img.shields.io/badge/AstrBot-%3E%3D4.27.0-f2994a.svg)](https://github.com/AstrBotDevs/AstrBot)
[![许可证](https://img.shields.io/badge/许可证-MIT-27ae60.svg)](LICENSE)

ComfyUI AI 绘画台是一个适用于 AstrBot 的本地 ComfyUI 绘图插件。它基于用户配置的 ComfyUI API 工作流，提供文生图、图生图和高清放大，并把模型、LoRA、提示词预设、画师串和 LLM 绘图整合到统一的指令和 WebUI 中。

## 功能特性

- 文生图、图生图和高清放大三种绘图模式。
- 基于原始 ComfyUI 工作流生成独立的 API 工作流副本。
- 支持核心模型、采样步数、CFG、种子、尺寸和重绘幅度等参数。
- 支持多个 LoRA，能够设置权重、中文昵称、多个旧指令简称、分类和启用状态。
- 读取 LoRA 的 CivitAI 触发词，并在启用或临时调用 LoRA 时加入绘图提示词；普通 CivitAI tag 不会自动进入提示词。
- 每个 LoRA 独立管理多个指令简称、分类和“指令简称 -> 预设内容”映射，完整模式和简洁模式都可以编辑；原有全局提示词预设继续保留。
- 支持 LoRA CivitAI 链接、触发词、预览图片、下载、上传和删除。
- 支持提示词预设和独立的画师串预设。
- 绘图指令可以同时使用提示词预设和 LoRA 简称；临时 LoRA 只在当前任务中生效。
- 支持使用 AstrBot 当前 AI 或插件独立 AI 优化提示词。
- LLM 绘图可选择使用 AstrBot LLM 提取结果，或由插件 AI 根据用户原话重新生成完整提示词；开发者模式可查看输入与最终提示词。
- LLM 绘图提示词和普通指令翻译分别使用独立的插件 AI 提示词，可在 WebUI 单独修改。
- 支持普通中文提示词的非 AI 翻译，翻译失败时保留原文继续绘图。
- 支持 AstrBot LLM 通过自然语言调用绘图工具。
- 支持自定义开始绘图提示和完成提示，可使用 AstrBot 当前人格回复。
- 支持普通图片消息和群聊合并转发两种发送方式。
- WebUI 分组管理工作流、AI、提示词预设、画师串和 LoRA。
- 支持在 WebUI 查看 ComfyUI 状态、模型目录、LoRA 目录和工作流目录。
- 支持上传和切换 API 工作流，明暗主题可切换。
- CivitAI LoRA 下载支持实时进度显示，下载完成后自动刷新模型和 LoRA 列表。
- CivitAI 下载支持模型页、版本页和直接下载链接；可选填写 API Key，自动处理 CDN 重定向、临时限流和断线重试。
- `/helpd` 会发送包含常用指令的中文帮助图片。

## 安装

### 从插件市场安装

在 AstrBot WebUI 的插件市场搜索 `ComfyUI AI 绘画台`，安装后重载插件或重启 AstrBot。

### 从 GitHub 安装

仓库地址：

<https://github.com/fornever123/astrbot_plugin_comfyui_studio>

将仓库目录复制到 AstrBot 的 `data/plugins/` 目录：

```text
AstrBot/data/plugins/astrbot_plugin_comfyui_ai_studio/
```

也可以下载仓库压缩包，将解压后的插件目录放入上述路径。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

安装完成后，在 AstrBot WebUI 重载插件，或重启 AstrBot。

## 使用方法

### 绘图指令

```text
/文生图 一名站在海边的少女
/文生图 夏空 海边
/图生图 改成夜景
/高清放大
```

图生图需要在同一条消息中附图或回复图片。高清放大支持附图、回复图片，也可以使用当前会话最近生成的图片。

### 参数

```text
/文生图 夏空 1号lora 海边 --步数=30 --cfg=5 --种子=-1
/文生图 ai 夏空 海边
/文生图 --模型=模型文件名 --负面=不需要的内容
/图生图 变成夜景 --强度=0.6
/高清放大 --放大=2 --强度=0.25
```

`ai` 开启当前任务的 AI 提示词优化；`noai` 可以关闭。预设和 LoRA 简称会在提示词处理前解析，因此可以同时生效：

```text
/文生图 夏空 1号lora 海边站立
```

上例会同时使用名为“夏空”的提示词预设和简称为“1号lora”的 LoRA。临时 LoRA 任务完成后不会改变默认启用列表。

### 模型、LoRA 和预设

```text
/模型 列表
/模型 模型文件名
/lora
/loraon 1号lora
/lora 添加 1号lora,2号lora
/lora 删除 1号lora
/lora 1号lora
/lora c站
/预设 列表
/预设 添加 夏空=ciaccona
/预设 修改 夏空=ciaccona, red hair
/预设 删除 夏空
/画师串 列表
/画师串 使用 画风001
```

`/lora` 查看全部 LoRA 及启用状态；`/lora 简称` 查看对应的 CivitAI 信息；`/lora c站` 查看全部 LoRA 的 CivitAI 信息。

### 工作流和状态

```text
/工作流 列表
/工作流 文生图 文件名
/工作流 图生图 文件名
/工作流 高清放大 文件名
/画图配置
/comfy状态
/helpd
```

### 自然语言绘图

启用 AstrBot LLM 工具后，可以直接发送自然语言：

```text
帮我画一张夏空在海边的图片，使用 1号lora
```

AstrBot 会根据请求选择绘图模式，并把识别出的预设、LoRA 简称和绘图参数传给插件。LLM 绘图任务会在后台执行，任务提交后可以继续聊天。

## WebUI 配置

打开 AstrBot WebUI 的插件配置页面：

- **工作流**：配置三种绘图模式的 API 工作流，上传或切换工作流。
- **AI**：配置 OpenAI 兼容服务地址、API Key 和模型，并测试模型列表和连接。
- **提示词与回复**：配置默认正面提示词、默认负面提示词、开始提示、完成提示、回复方式和合并转发。
- **提示词预设**：新增、修改和删除提示词预设。
- **画师串预设**：维护和切换画师串预设，默认预设为 `画风001`。
- **LoRA 管理**：管理模型文件、昵称、多个指令简称、权重、分类、专属预设、CivitAI 信息和预览图，并支持搜索。

默认配置示例：

```text
ComfyUI 地址：http://127.0.0.1:8188
默认宽度：832
默认高度：1216
默认步数：30
默认 CFG：5
```

## 工作流要求

插件不会覆盖原始工作流文件。首次加载时，会根据配置中的原始工作流生成插件目录下的 API 工作流副本：

```text
workflows/文生图.json
workflows/图生图.json
workflows/高清放大.json
```

原始工作流使用的自定义节点必须安装在 ComfyUI 中。常见依赖包括 rgthree、Crystools、Anima 相关节点和 Ultimate SD Upscale。核心模型、LoRA 和放大模型也必须存在于 ComfyUI 对应的模型目录。

## 故障排查

### ComfyUI 未连接

确认 ComfyUI 已启动，并检查插件中的 ComfyUI 地址。可以使用：

```text
/comfy状态
```

### 模型或 LoRA 不存在

使用 `/画图配置` 查看模型目录和 LoRA 目录，确认文件已经放入 ComfyUI 的对应目录，然后在 WebUI 刷新模型列表。

### 工作流节点缺失

查看 ComfyUI 控制台中的节点名称，安装原始工作流所需的自定义节点。插件只能提交工作流，不能替 ComfyUI 安装缺少的节点。

### 生成黑图或失败

先在 ComfyUI WebUI 中单独运行对应 API 工作流，确认模型、VAE、采样器、放大模型和自定义节点均可用，再通过 `/工作流` 选择正确的工作流文件。

## 数据和隐私

插件运行数据保存在 AstrBot 的插件数据目录，不会写入公开仓库。API Key 只用于请求配置的 AI 服务，不会通过状态接口返回到浏览器。

## 开源协议

本项目采用 [MIT License](LICENSE) 开源。
