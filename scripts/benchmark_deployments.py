#!/usr/bin/env python3
"""Benchmark qwen3-1.7B deployments across accuracy, speed, and resource usage.

This script compares three OpenAI-compatible endpoints:
- vllm_gpu
- vllm_cpu
- llamacpp_cpu
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


ACCURACY_CASES = [
    {"id": "math_01", "question": "27 + 58 = ?", "answer": "85"},
    {"id": "math_02", "question": "144 / 12 = ?", "answer": "12"},
    {"id": "math_03", "question": "39 * 4 = ?", "answer": "156"},
    {"id": "math_04", "question": "1000 - 333 = ?", "answer": "667"},
    {"id": "math_05", "question": "2 的 10 次方等于多少？", "answer": "1024"},
    {"id": "math_06", "question": "15 * 15 = ?", "answer": "225"},
    {"id": "math_07", "question": "(18 + 24) / 6 = ?", "answer": "7"},
    {"id": "math_08", "question": "90 / 5 + 3 = ?", "answer": "21"},
    {"id": "math_09", "question": "13 + 29 + 31 = ?", "answer": "73"},
    {"id": "math_10", "question": "81 的平方根（正数）是多少？", "answer": "9"},
]


@dataclass
class Deployment:
    name: str
    base_url: str
    model: str
    container_name: str
    gpu_index: int | None = None


@dataclass
class RequestResult:
    case_id: str
    ok: bool
    latency_s: float
    completion_tokens: int
    token_speed_tps: float
    score: float
    response_text: str
    error: str | None
    cpu_avg_pct: float | None
    cpu_peak_pct: float | None
    mem_avg_mib: float | None
    mem_peak_mib: float | None


class DockerSampler:
    def __init__(self, container_names: list[str], interval_s: float = 1.0) -> None:
        self.container_names = set(container_names)
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            ts = time.time()
            data = self._docker_stats_once()
            if data is not None:
                for row in data:
                    if row["name"] in self.container_names:
                        row["ts"] = ts
                        self.samples.append(row)
            time.sleep(self.interval_s)

    def _docker_stats_once(self) -> list[dict[str, Any]] | None:
        cmd = [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            self.enabled = False
            return None

        rows: list[dict[str, Any]] = []
        for line in out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            name, cpu_s, mem_s = parts
            rows.append(
                {
                    "name": name.strip(),
                    "cpu_pct": parse_percent(cpu_s),
                    "mem_mib": parse_mem_mib(mem_s),
                }
            )
        return rows

    def summarize_interval(self, container_name: str, start_ts: float, end_ts: float) -> dict[str, float | None]:
        # Short requests can complete between two sampling ticks; include one-tick margin.
        margin = max(self.interval_s, 0.05)
        interval = [
            s
            for s in self.samples
            if s["name"] == container_name and (start_ts - margin) <= float(s["ts"]) <= (end_ts + margin)
        ]
        if not interval:
            return {
                "cpu_avg_pct": None,
                "cpu_peak_pct": None,
                "mem_avg_mib": None,
                "mem_peak_mib": None,
            }

        cpu_values = [float(s["cpu_pct"]) for s in interval]
        mem_values = [float(s["mem_mib"]) for s in interval]
        return {
            "cpu_avg_pct": statistics.mean(cpu_values),
            "cpu_peak_pct": max(cpu_values),
            "mem_avg_mib": statistics.mean(mem_values),
            "mem_peak_mib": max(mem_values),
        }

    def snapshot_container(self, container_name: str) -> dict[str, float] | None:
        data = self._docker_stats_once()
        if data is None:
            return None
        for row in data:
            if row["name"] == container_name:
                return {
                    "cpu_pct": float(row["cpu_pct"]),
                    "mem_mib": float(row["mem_mib"]),
                }
        return None


class NvidiaSampler:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            ts = time.time()
            data = self._nvidia_smi_once()
            if data is not None:
                for row in data:
                    row["ts"] = ts
                    self.samples.append(row)
            time.sleep(self.interval_s)

    def _nvidia_smi_once(self) -> list[dict[str, float]] | None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            self.enabled = False
            return None

        rows: list[dict[str, float]] = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                rows.append(
                    {
                        "gpu_index": int(parts[0]),
                        "gpu_util_pct": float(parts[1]),
                        "gpu_mem_mib": float(parts[2]),
                    }
                )
            except ValueError:
                continue
        return rows

    def summarize_interval(self, gpu_index: int, start_ts: float, end_ts: float) -> dict[str, float | None]:
        margin = max(self.interval_s, 0.05)
        interval = [
            s
            for s in self.samples
            if int(s["gpu_index"]) == gpu_index and (start_ts - margin) <= float(s["ts"]) <= (end_ts + margin)
        ]
        if not interval:
            return {
                "gpu_avg_util_pct": None,
                "gpu_peak_util_pct": None,
                "gpu_avg_mem_mib": None,
                "gpu_peak_mem_mib": None,
            }
        util_values = [float(s["gpu_util_pct"]) for s in interval]
        mem_values = [float(s["gpu_mem_mib"]) for s in interval]
        return {
            "gpu_avg_util_pct": statistics.mean(util_values),
            "gpu_peak_util_pct": max(util_values),
            "gpu_avg_mem_mib": statistics.mean(mem_values),
            "gpu_peak_mem_mib": max(mem_values),
        }


def parse_percent(value: str) -> float:
    return float(value.strip().replace("%", "") or "0")


def parse_mem_mib(value: str) -> float:
    used = value.split("/")[0].strip()
    m = re.match(r"([0-9.]+)\s*([KMGTP]i?)B", used, flags=re.IGNORECASE)
    if not m:
        return 0.0
    num = float(m.group(1))
    unit = m.group(2).upper()
    factors = {
        "KB": 1.0 / 1024,
        "KI": 1.0 / 1024,
        "KIB": 1.0 / 1024,
        "MB": 1.0,
        "MI": 1.0,
        "MIB": 1.0,
        "GB": 1024.0,
        "GI": 1024.0,
        "GIB": 1024.0,
        "TB": 1024.0 * 1024.0,
        "TI": 1024.0 * 1024.0,
        "TIB": 1024.0 * 1024.0,
        "PB": 1024.0 * 1024.0 * 1024.0,
        "PI": 1024.0 * 1024.0 * 1024.0,
        "PIB": 1024.0 * 1024.0 * 1024.0,
    }
    return num * factors.get(unit, 1.0)


def merge_interval_and_snapshots(
    interval_usage: dict[str, float | None],
    snap_before: dict[str, float] | None,
    snap_after: dict[str, float] | None,
) -> dict[str, float | None]:
    if interval_usage["cpu_avg_pct"] is not None and interval_usage["mem_avg_mib"] is not None:
        return interval_usage

    snaps = [s for s in [snap_before, snap_after] if s is not None]
    if not snaps:
        return interval_usage

    cpu_values = [float(s["cpu_pct"]) for s in snaps]
    mem_values = [float(s["mem_mib"]) for s in snaps]
    return {
        "cpu_avg_pct": statistics.mean(cpu_values),
        "cpu_peak_pct": max(cpu_values),
        "mem_avg_mib": statistics.mean(mem_values),
        "mem_peak_mib": max(mem_values),
    }


def normalize_answer(text: str) -> str:
    # Prefer explicit format "答案: ...", fallback to last number-like token.
    m = re.search(r"答案\s*[:：]\s*([^\n]+)", text)
    if m:
        return m.group(1).strip().lower()

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip().lower()
    return text.strip().lower()


def build_prompt(question: str) -> str:
    return (
        "请直接给出最终答案，不要解释。"
        "输出格式必须是：答案: <你的答案>\n"
        f"题目：{question}"
    )


def apply_no_think_prefix(
    prompt: str,
    deployment_name: str,
    no_think_for_llamacpp: bool,
    no_think_for_vllm: bool,
) -> str:
    if deployment_name == "llamacpp_cpu" and no_think_for_llamacpp:
        return "/no_think " + prompt
    if deployment_name in {"vllm_cpu", "vllm_gpu"} and no_think_for_vllm:
        return "/no_think " + prompt
    return prompt


def post_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def extract_text_and_tokens(resp_json: dict[str, Any]) -> tuple[str, int]:
    text = ""
    try:
        text = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        text = ""

    completion_tokens = 0
    usage = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
    if isinstance(usage, dict):
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    if completion_tokens <= 0:
        completion_tokens = max(1, len(text.split()))
    return text, completion_tokens


def run_case(
    deployment: Deployment,
    case: dict[str, str],
    timeout_s: int,
    max_tokens: int,
    sampler: DockerSampler,
    no_think_for_llamacpp: bool,
    no_think_for_vllm: bool,
) -> RequestResult:
    prompt = apply_no_think_prefix(
        prompt=build_prompt(case["question"]),
        deployment_name=deployment.name,
        no_think_for_llamacpp=no_think_for_llamacpp,
        no_think_for_vllm=no_think_for_vllm,
    )

    snap_before = sampler.snapshot_container(deployment.container_name)
    t0 = time.time()
    try:
        resp_json = post_chat_completion(
            base_url=deployment.base_url,
            model=deployment.model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        t1 = time.time()
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, ValueError) as exc:
        t1 = time.time()
        snap_after = sampler.snapshot_container(deployment.container_name)
        res_usage = sampler.summarize_interval(deployment.container_name, t0, t1)
        res_usage = merge_interval_and_snapshots(res_usage, snap_before, snap_after)
        return RequestResult(
            case_id=case["id"],
            ok=False,
            latency_s=max(t1 - t0, 0.0),
            completion_tokens=0,
            token_speed_tps=0.0,
            score=0.0,
            response_text="",
            error=str(exc),
            cpu_avg_pct=res_usage["cpu_avg_pct"],
            cpu_peak_pct=res_usage["cpu_peak_pct"],
            mem_avg_mib=res_usage["mem_avg_mib"],
            mem_peak_mib=res_usage["mem_peak_mib"],
        )

    text, completion_tokens = extract_text_and_tokens(resp_json)
    latency = max(t1 - t0, 1e-9)
    answer = normalize_answer(text)
    expected = case["answer"].strip().lower()
    score = 1.0 if answer == expected else 0.0
    snap_after = sampler.snapshot_container(deployment.container_name)
    usage = sampler.summarize_interval(deployment.container_name, t0, t1)
    usage = merge_interval_and_snapshots(usage, snap_before, snap_after)

    return RequestResult(
        case_id=case["id"],
        ok=True,
        latency_s=latency,
        completion_tokens=completion_tokens,
        token_speed_tps=completion_tokens / latency,
        score=score,
        response_text=text,
        error=None,
        cpu_avg_pct=usage["cpu_avg_pct"],
        cpu_peak_pct=usage["cpu_peak_pct"],
        mem_avg_mib=usage["mem_avg_mib"],
        mem_peak_mib=usage["mem_peak_mib"],
    )


def summarize(
    results: list[RequestResult],
    deployment_usage: dict[str, float | None] | None = None,
    gpu_usage: dict[str, float | None] | None = None,
) -> dict[str, float]:
    if not results:
        return {
            "accuracy": 0.0,
            "avg_latency_s": 0.0,
            "p50_latency_s": 0.0,
            "avg_token_speed_tps": 0.0,
            "avg_cpu_pct": 0.0,
            "peak_cpu_pct": 0.0,
            "avg_mem_mib": 0.0,
            "peak_mem_mib": 0.0,
            "success_rate": 0.0,
            "avg_gpu_util_pct": 0.0,
            "peak_gpu_util_pct": 0.0,
            "avg_vram_mib": 0.0,
            "peak_vram_mib": 0.0,
        }

    def non_null(values: list[float | None]) -> list[float]:
        return [float(v) for v in values if v is not None]

    latencies = [r.latency_s for r in results]
    speed = [r.token_speed_tps for r in results if r.token_speed_tps > 0]
    cpu_avg = non_null([r.cpu_avg_pct for r in results])
    cpu_peak = non_null([r.cpu_peak_pct for r in results])
    mem_avg = non_null([r.mem_avg_mib for r in results])
    mem_peak = non_null([r.mem_peak_mib for r in results])

    out = {
        "accuracy": statistics.mean(r.score for r in results),
        "avg_latency_s": statistics.mean(latencies),
        "p50_latency_s": statistics.median(latencies),
        "avg_token_speed_tps": statistics.mean(speed) if speed else 0.0,
        "avg_cpu_pct": statistics.mean(cpu_avg) if cpu_avg else 0.0,
        "peak_cpu_pct": max(cpu_peak) if cpu_peak else 0.0,
        "avg_mem_mib": statistics.mean(mem_avg) if mem_avg else 0.0,
        "peak_mem_mib": max(mem_peak) if mem_peak else 0.0,
        "success_rate": statistics.mean(1.0 if r.ok else 0.0 for r in results),
        "avg_gpu_util_pct": 0.0,
        "peak_gpu_util_pct": 0.0,
        "avg_vram_mib": 0.0,
        "peak_vram_mib": 0.0,
    }

    if deployment_usage is not None:
        if deployment_usage.get("cpu_avg_pct") is not None:
            out["avg_cpu_pct"] = float(deployment_usage["cpu_avg_pct"])
        if deployment_usage.get("cpu_peak_pct") is not None:
            out["peak_cpu_pct"] = float(deployment_usage["cpu_peak_pct"])
        if deployment_usage.get("mem_avg_mib") is not None:
            out["avg_mem_mib"] = float(deployment_usage["mem_avg_mib"])
        if deployment_usage.get("mem_peak_mib") is not None:
            out["peak_mem_mib"] = float(deployment_usage["mem_peak_mib"])

    if gpu_usage is not None:
        if gpu_usage.get("gpu_avg_util_pct") is not None:
            out["avg_gpu_util_pct"] = float(gpu_usage["gpu_avg_util_pct"])
        if gpu_usage.get("gpu_peak_util_pct") is not None:
            out["peak_gpu_util_pct"] = float(gpu_usage["gpu_peak_util_pct"])
        if gpu_usage.get("gpu_avg_mem_mib") is not None:
            out["avg_vram_mib"] = float(gpu_usage["gpu_avg_mem_mib"])
        if gpu_usage.get("gpu_peak_mem_mib") is not None:
            out["peak_vram_mib"] = float(gpu_usage["gpu_peak_mem_mib"])

    return out


def make_markdown_report(summary_by_dep: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Qwen3-1.7B Deployment Benchmark",
        "",
        "| Deployment | Accuracy | Success | Avg Latency (s) | P50 Latency (s) | Avg Token/s | Avg CPU (%) | Peak CPU (%) | Avg Mem (MiB) | Peak Mem (MiB) | Avg GPU Util (%) | Peak GPU Util (%) | Avg VRAM (MiB) | Peak VRAM (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summary_by_dep.items():
        lines.append(
            "| {name} | {accuracy:.2%} | {success_rate:.2%} | {avg_latency_s:.3f} | {p50_latency_s:.3f} | "
            "{avg_token_speed_tps:.2f} | {avg_cpu_pct:.2f} | {peak_cpu_pct:.2f} | {avg_mem_mib:.1f} | {peak_mem_mib:.1f} | "
            "{avg_gpu_util_pct:.2f} | {peak_gpu_util_pct:.2f} | {avg_vram_mib:.1f} | {peak_vram_mib:.1f} |".format(
                name=name,
                **s,
            )
        )
    lines.append("")
    lines.append("Accuracy is exact-match on arithmetic QA with enforced answer format.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark vLLM/llama.cpp deployments for qwen3-1.7B")
    p.add_argument(
        "--deployments",
        nargs="+",
        choices=["vllm_gpu", "vllm_cpu", "llamacpp_cpu"],
        default=["vllm_gpu", "vllm_cpu", "llamacpp_cpu"],
        help="Subset of deployments to benchmark",
    )
    p.add_argument("--requests-per-deployment", type=int, default=len(ACCURACY_CASES))
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--resource-sample-interval", type=float, default=0.2)
    p.add_argument("--resource-min-window", type=float, default=2.0)
    p.add_argument("--output-dir", default="benchmark_results")
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument(
            "--llamacpp-no-think",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Prefix llama.cpp prompts with /no_think (default: true)",
        )
        p.add_argument(
            "--vllm-no-think",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Prefix vLLM prompts with /no_think (default: true)",
        )
    else:
        p.add_argument(
            "--llamacpp-no-think",
            dest="llamacpp_no_think",
            action="store_true",
            default=True,
            help="Prefix llama.cpp prompts with /no_think (default: true)",
        )
        p.add_argument(
            "--no-llamacpp-no-think",
            dest="llamacpp_no_think",
            action="store_false",
            help="Disable /no_think prefix for llama.cpp prompts",
        )
        p.add_argument(
            "--vllm-no-think",
            dest="vllm_no_think",
            action="store_true",
            default=True,
            help="Prefix vLLM prompts with /no_think (default: true)",
        )
        p.add_argument(
            "--no-vllm-no-think",
            dest="vllm_no_think",
            action="store_false",
            help="Disable /no_think prefix for vLLM prompts",
        )

    p.add_argument("--vllm-gpu-url", default="http://127.0.0.1:8000")
    p.add_argument("--vllm-gpu-model", default="qwen3-1.7b")
    p.add_argument("--vllm-gpu-container", default="vllm-gpu-qwen")
    p.add_argument("--vllm-gpu-index", type=int, default=0)

    p.add_argument("--vllm-cpu-url", default="http://127.0.0.1:8001")
    p.add_argument("--vllm-cpu-model", default="qwen3-1.7b-cpu")
    p.add_argument("--vllm-cpu-container", default="vllm-cpu-qwen")

    p.add_argument("--llamacpp-url", default="http://127.0.0.1:8010")
    p.add_argument("--llamacpp-model", default="llama-cpu")
    p.add_argument("--llamacpp-container", default="llamacpp-cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    deployments_all = [
        Deployment("vllm_gpu", args.vllm_gpu_url, args.vllm_gpu_model, args.vllm_gpu_container, args.vllm_gpu_index),
        Deployment("vllm_cpu", args.vllm_cpu_url, args.vllm_cpu_model, args.vllm_cpu_container),
        Deployment("llamacpp_cpu", args.llamacpp_url, args.llamacpp_model, args.llamacpp_container),
    ]
    deployments = [d for d in deployments_all if d.name in set(args.deployments)]

    cases = ACCURACY_CASES[: max(1, min(args.requests_per_deployment, len(ACCURACY_CASES)))]

    sampler = DockerSampler([d.container_name for d in deployments], interval_s=args.resource_sample_interval)
    gpu_sampler = NvidiaSampler(interval_s=args.resource_sample_interval)
    sampler.start()
    gpu_sampler.start()

    all_results: dict[str, list[RequestResult]] = {d.name: [] for d in deployments}
    deployment_usage_map: dict[str, dict[str, float | None]] = {}
    deployment_gpu_usage_map: dict[str, dict[str, float | None]] = {}

    try:
        for dep in deployments:
            print(f"\n=== Benchmarking {dep.name} ({dep.base_url}) ===")

            # Warmup request is excluded from summary.
            try:
                warmup_prompt = apply_no_think_prefix(
                    prompt="请回复：答案: warmup",
                    deployment_name=dep.name,
                    no_think_for_llamacpp=args.llamacpp_no_think,
                    no_think_for_vllm=args.vllm_no_think,
                )
                _ = post_chat_completion(
                    base_url=dep.base_url,
                    model=dep.model,
                    prompt=warmup_prompt,
                    max_tokens=16,
                    timeout_s=args.timeout,
                )
                print("Warmup: ok")
            except Exception as exc:  # noqa: BLE001
                print(f"Warmup failed: {exc}")

            dep_start_ts = time.time()
            for i, case in enumerate(cases, start=1):
                result = run_case(
                    deployment=dep,
                    case=case,
                    timeout_s=args.timeout,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    no_think_for_llamacpp=args.llamacpp_no_think,
                    no_think_for_vllm=args.vllm_no_think,
                )
                all_results[dep.name].append(result)
                status = "OK" if result.ok else "ERR"
                print(
                    f"[{dep.name}] {i}/{len(cases)} {case['id']} {status} "
                    f"lat={result.latency_s:.2f}s tps={result.token_speed_tps:.2f} score={result.score:.0f}"
                )
            dep_end_ts = time.time()

            # For very short benchmarks, widen summary window to improve sampling stability.
            if (dep_end_ts - dep_start_ts) < args.resource_min_window:
                pad = (args.resource_min_window - (dep_end_ts - dep_start_ts)) / 2
                dep_start_ts -= pad
                dep_end_ts += pad

            deployment_usage_map[dep.name] = sampler.summarize_interval(dep.container_name, dep_start_ts, dep_end_ts)
            if dep.gpu_index is not None:
                deployment_gpu_usage_map[dep.name] = gpu_sampler.summarize_interval(dep.gpu_index, dep_start_ts, dep_end_ts)
            else:
                deployment_gpu_usage_map[dep.name] = {
                    "gpu_avg_util_pct": None,
                    "gpu_peak_util_pct": None,
                    "gpu_avg_mem_mib": None,
                    "gpu_peak_mem_mib": None,
                }
    finally:
        sampler.stop()
        gpu_sampler.stop()

    summary_by_dep: dict[str, dict[str, float]] = {
        dep.name: summarize(
            all_results[dep.name],
            deployment_usage=deployment_usage_map.get(dep.name),
            gpu_usage=deployment_gpu_usage_map.get(dep.name),
        )
        for dep in deployments
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"benchmark_{ts}.json"
    md_path = out_dir / f"benchmark_{ts}.md"

    json_payload = {
        "created_at": datetime.now().isoformat(),
        "args": vars(args),
        "summary": summary_by_dep,
        "details": {
            name: [asdict(r) for r in rows] for name, rows in all_results.items()
        },
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_report = make_markdown_report(summary_by_dep)
    md_path.write_text(md_report + "\n", encoding="utf-8")

    print("\n" + md_report)
    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if not sampler.enabled:
        print("Resource sampling note: docker stats unavailable; CPU/Mem fields may be zero.")
    if not gpu_sampler.enabled:
        print("GPU sampling note: nvidia-smi unavailable; GPU/VRAM fields may be zero.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
