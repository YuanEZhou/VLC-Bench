# 推理基准对比：vLLM vs llama.cpp

一个用于对比 `vllm_gpu`、`vllm_cpu`、`llamacpp_cpu` 的小型基准测试项目，主要关注：

- 推理速度（延迟、Token/s）
- 资源占用（CPU、内存、GPU 利用率、显存）
- 工具调用能力（Tool Calling 成功率与准确率）

当前测试模型以 **Qwen3-1.7B** 为主。

## 项目结构

```text
scripts/
  start_vllm_gpu.sh         # 启动 vLLM GPU 容器（OpenAI 兼容接口）
  start_vllm_cpu.sh         # 启动 vLLM CPU 容器（OpenAI 兼容接口）
  start_llamacpp_cpu.sh     # 启动 llama.cpp CPU 容器（OpenAI 兼容接口）
  benchmark_deployments.py  # 基础能力/速度/资源占用基准
  benchmark_tool_calls.py   # 工具调用能力基准
```

## 环境要求

- Linux
- Docker
- Python 3.10+
- （可选）NVIDIA GPU + `nvidia-smi`（用于采集 GPU 指标）

## 快速开始

### 1) 启动三个服务

在仓库根目录执行：

```bash
bash scripts/start_vllm_gpu.sh
bash scripts/start_vllm_cpu.sh
bash scripts/start_llamacpp_cpu.sh
```

默认端口：

- `vllm_gpu`: `http://127.0.0.1:8000`
- `vllm_cpu`: `http://127.0.0.1:8001`
- `llamacpp_cpu`: `http://127.0.0.1:8010`

### 2) 运行基础性能/资源 benchmark

```bash
python scripts/benchmark_deployments.py
```

输出：

- `benchmark_results/benchmark_YYYYMMDD_HHMMSS.json`
- `benchmark_results/benchmark_YYYYMMDD_HHMMSS.md`

### 3) 运行工具调用 benchmark

```bash
python scripts/benchmark_tool_calls.py
```

输出：

- `benchmark_results/toolcall_benchmark_YYYYMMDD_HHMMSS.json`
- `benchmark_results/toolcall_benchmark_YYYYMMDD_HHMMSS.md`

## 常用参数示例

只测部分部署：

```bash
python scripts/benchmark_deployments.py --deployments vllm_gpu vllm_cpu
python scripts/benchmark_tool_calls.py --deployments vllm_gpu llamacpp_cpu
```

控制请求数：

```bash
python scripts/benchmark_deployments.py --requests-per-deployment 5
python scripts/benchmark_tool_calls.py --requests-per-deployment 4
```

## 结果说明

### `benchmark_deployments.py`

主要指标：

- `accuracy`: 算术题精确匹配正确率
- `avg_latency_s` / `p50_latency_s`: 平均/P50 延迟
- `avg_token_speed_tps`: 平均生成速度（Token/s）
- `avg_cpu_pct` / `peak_cpu_pct`: CPU 占用
- `avg_mem_mib` / `peak_mem_mib`: 内存占用
- `avg_gpu_util_pct` / `peak_gpu_util_pct`: GPU 利用率
- `avg_vram_mib` / `peak_vram_mib`: 显存占用

### `benchmark_tool_calls.py`

主要指标：

- `overall_success_rate`: 工具调用链路整体成功率
- `tool_call_rate`: 是否触发工具调用
- `correct_tool_rate`: 工具名是否正确
- `correct_args_rate`: 工具参数是否正确
- `correct_final_answer_rate`: 最终答案是否正确

## 注意事项

- 脚本默认基于 OpenAI 兼容接口 `/v1/chat/completions`。
- 若无法采集资源数据（如 `docker stats` 或 `nvidia-smi` 不可用），对应指标可能为 0。
- 你可以通过环境变量覆盖启动脚本中的参数（例如端口、模型路径、容器名等）。


