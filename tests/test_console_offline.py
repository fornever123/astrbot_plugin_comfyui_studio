"""Exercise the actual console loading flow without a ComfyUI connection."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_unconnected_comfyui_does_not_break_first_time_configuration():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to exercise the console loading flow")
    source = Path(__file__).resolve().parents[1] / "app.js"
    program = r"""
const fs = require('fs'), vm = require('vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const load = source.slice(source.indexOf('async function load() {'), source.indexOf('\nfunction fillConfig()'));
const elements = {version: {}, status: {}, statusText: {}, model: {value: ''}};
const sandbox = {
  API: '', config: {}, modelData: {},
  document: {getElementById: id => elements[id] || null},
  get: async path => {
    if (path !== '/status') throw new Error('Unexpected request: ' + path);
    return {version: 'test', comfy: {ok: false, error: '连接被拒绝'}, config: {model_name: 'my-model'}};
  },
  initTheme() {}, initLoraMode() {}, initLoraSection() {}, initEngineLoras() {}, initViews() {},
  fillConfig() {elements.model.value = sandbox.config.model_name;},
  loadModels: async () => {throw new Error('离线模型接口不可用');},
  loadWorkflows: async () => {}, loadPresets: async () => {},
  loadArtistPresets: async () => {}, loadPathSettings: async () => {},
  setSelectOptions() {}, show() {},
};
vm.createContext(sandbox);
vm.runInContext(load, sandbox);
sandbox.load().then(() => {
  if (elements.status.textContent !== '未连接') throw new Error('Console failed: ' + elements.status.textContent);
  if (elements.model.value !== 'my-model') throw new Error('Existing model was lost');
  if (!elements.statusText.textContent.includes('连接被拒绝')) throw new Error('Connection error was hidden');
}).catch(error => {console.error(error); process.exitCode = 1;});
"""
    result = subprocess.run(
        [node, "-e", program, str(source)],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
