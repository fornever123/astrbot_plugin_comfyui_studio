# Anima ComfyUI 桌面端

这是插件附带的独立桌面工具。它不占用 AstrBot WebUI，使用 OpenAI 兼容 API 进行对话，并直接通过 ComfyUI API 提交任务。

## 启动

在插件根目录打开 PowerShell：

```powershell
python -m pip install -r desktop\requirements.txt
python desktop\desktop_app.py
```

Windows 也可以双击根目录的 `启动桌面端.bat`。

桌面端使用 Python 自带的 Tkinter，不需要 Electron。首次运行会在 `%APPDATA%\AnimaComfyUIStudio\desktop.json` 保存设置和工作流映射。

## 第一次配置

1. 打开“设置”。
2. 填写 ComfyUI 地址，默认是 `http://127.0.0.1:8188`。
3. 填写 OpenAI 兼容服务地址、API Key 和模型，点击“获取 AI 模型列表”和“测试 AI 连接”。
4. 打开“工作流”，选择 API 格式或 ComfyUI 编辑器格式 JSON。
5. 点击“AI 一键适配”。AI 只会返回节点输入映射，原工作流文件不会被改写。
6. 如果 AI 暂时不可用，可以点击“离线自动适配”；识别结果可以在右侧 JSON 中检查和修改，再点击“保存映射 JSON”。
7. 在“文生图”或“图生图”页选择工作流并开始绘图。

## 支持范围

通用适配器会尝试识别正面/负面提示词、LoadImage 参考图、核心模型、文本编码器、VAE、LoRA、采样参数、画布尺寸和 SaveImage 输出节点。未知节点、节点连接和原有参数会保留。

“任意工作流”仍然取决于工作流本身提供可调用的输入节点和输出节点。对于完全自定义的节点，使用“工作流”页编辑映射即可；如果节点类型本身没有安装在 ComfyUI 中，桌面端无法替用户安装该节点。

图生图最多上传三张图片，实际接入数量由工作流中的参考图输入数量决定。桌面端会在提交前上传图片，并在“ComfyUI”页显示最后一次实际提交的 API 工作流。

## 与 AstrBot 插件的关系

AstrBot 插件原有指令、LLM 工具和 WebUI 保持不变。桌面端是额外入口，使用独立设置文件和独立输出目录；两者可以连接同一个 ComfyUI 服务，但不会共享正在执行的桌面端对话历史。
