#!/usr/bin/env python3
"""Benchmark tool-calling capability across three OpenAI-compatible endpoints.

This script compares:
- vllm_gpu
- vllm_cpu
- llamacpp_cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


@dataclass
class Deployment:
    name: str
    base_url: str
    model: str


@dataclass
class ToolCase:
    id: str
    prompt: str
    expected_tool: str
    expected_args: dict[str, Any]
    expected_answer_hint: str


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    first_call_latency_s: float
    total_latency_s: float
    completion_tokens: int
    token_speed_tps: float
    has_tool_call: bool
    correct_tool: bool
    correct_args: bool
    correct_final_answer: bool
    tool_name: str | None
    tool_args: dict[str, Any] | None
    final_text: str
    error: str | None


TOOL_CASES = [
    ToolCase(
        id="tool_01_add",
        prompt="请使用工具计算 19 + 23，并给出最终答案。",
        expected_tool="add_numbers",
        expected_args={"a": 19, "b": 23},
        expected_answer_hint="42",
    ),
    ToolCase(
        id="tool_02_temp",
        prompt="请使用工具把 25 摄氏度转换为华氏度，并给出最终答案。",
        expected_tool="celsius_to_fahrenheit",
        expected_args={"celsius": 25},
        expected_answer_hint="77",
    ),
    ToolCase(
        id="tool_03_reverse",
        prompt="请使用工具反转字符串 tool-calling，并给出最终答案。",
        expected_tool="reverse_text",
        expected_args={"text": "tool-calling"},
        expected_answer_hint="gnillac-loot",
    ),
    ToolCase(
        id="tool_04_add",
        prompt="请使用工具计算 -7 + 18，并给出最终答案。",
        expected_tool="add_numbers",
        expected_args={"a": -7, "b": 18},
        expected_answer_hint="11",
    ),
]


def build_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two integers and return their sum.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "celsius_to_fahrenheit",
                "description": "Convert celsius to fahrenheit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "celsius": {"type": "number"},
                    },
                    "required": ["celsius"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reverse_text",
                "description": "Reverse input text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def build_user_prompt(case: ToolCase) -> str:
    return (
        "/no_think 先调用一个最合适的工具来解决任务。"
        "这一轮不要直接输出最终答案。"
        f"任务：{case.prompt}"
    )


def post_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout_s: int,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {exc.code}: {err_body}") from exc
    except (ConnectionResetError, OSError):
        # Some servers reset the first connection right after startup; retry once.
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    return json.loads(body)


def get_message(resp_json: dict[str, Any]) -> dict[str, Any]:
    try:
        msg = resp_json["choices"][0]["message"]
        if isinstance(msg, dict):
            return msg
    except (KeyError, IndexError, TypeError):
        pass
    return {}


def completion_tokens(resp_json: dict[str, Any]) -> int:
    usage = resp_json.get("usage", {}) if isinstance(resp_json, dict) else {}
    if isinstance(usage, dict):
        return int(usage.get("completion_tokens", 0) or 0)
    return 0


def parse_tool_call(message: dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any] | None]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None, None, None

    first = tool_calls[0]
    if not isinstance(first, dict):
        return None, None, None

    tool_id = first.get("id")
    fn = first.get("function", {})
    if not isinstance(fn, dict):
        return None, None, None
    name = fn.get("name")
    raw_args = fn.get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else None
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        args = None
    return (str(tool_id) if tool_id is not None else None, str(name) if name is not None else None, args)


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 1e-9
    return expected == actual


def args_match(expected_args: dict[str, Any], actual_args: dict[str, Any] | None) -> bool:
    if actual_args is None:
        return False
    for key, expected_value in expected_args.items():
        if key not in actual_args:
            return False
        if not values_equal(expected_value, actual_args[key]):
            return False
    return True


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "add_numbers":
        a = int(tool_args["a"])
        b = int(tool_args["b"])
        return {"result": a + b}

    if tool_name == "celsius_to_fahrenheit":
        c = float(tool_args["celsius"])
        return {"result": c * 9.0 / 5.0 + 32.0}

    if tool_name == "reverse_text":
        text = str(tool_args["text"])
        return {"result": text[::-1]}

    raise ValueError(f"Unsupported tool: {tool_name}")


def run_case(
    deployment: Deployment,
    case: ToolCase,
    timeout_s: int,
    max_tokens: int,
    first_tool_choice: str,
) -> CaseResult:
    tools = build_tools()
    user_message = {"role": "user", "content": build_user_prompt(case)}

    t0 = time.time()
    try:
        first_resp = post_chat_completion(
            base_url=deployment.base_url,
            model=deployment.model,
            messages=[user_message],
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            tools=tools,
            tool_choice=first_tool_choice,
        )
        t1 = time.time()
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, ValueError, OSError) as exc:
        t1 = time.time()
        return CaseResult(
            case_id=case.id,
            ok=False,
            first_call_latency_s=max(t1 - t0, 0.0),
            total_latency_s=max(t1 - t0, 0.0),
            completion_tokens=0,
            token_speed_tps=0.0,
            has_tool_call=False,
            correct_tool=False,
            correct_args=False,
            correct_final_answer=False,
            tool_name=None,
            tool_args=None,
            final_text="",
            error=str(exc),
        )

    first_msg = get_message(first_resp)
    first_tokens = completion_tokens(first_resp)
    tool_id, tool_name, tool_args = parse_tool_call(first_msg)
    has_tool_call = tool_name is not None and tool_args is not None
    correct_tool = tool_name == case.expected_tool
    correct_args = args_match(case.expected_args, tool_args)

    if not has_tool_call:
        latency = max(t1 - t0, 1e-9)
        raw_text = str(first_msg.get("content") or "")
        answer_ok = case.expected_answer_hint in raw_text
        return CaseResult(
            case_id=case.id,
            ok=False,
            first_call_latency_s=latency,
            total_latency_s=latency,
            completion_tokens=max(1, first_tokens),
            token_speed_tps=max(1, first_tokens) / latency,
            has_tool_call=False,
            correct_tool=False,
            correct_args=False,
            correct_final_answer=answer_ok,
            tool_name=tool_name,
            tool_args=tool_args,
            final_text=raw_text,
            error="Model did not return valid tool_calls",
        )

    try:
        tool_result = execute_tool(tool_name, tool_args)
    except Exception as exc:  # noqa: BLE001
        latency = max(t1 - t0, 1e-9)
        return CaseResult(
            case_id=case.id,
            ok=False,
            first_call_latency_s=latency,
            total_latency_s=latency,
            completion_tokens=max(1, first_tokens),
            token_speed_tps=max(1, first_tokens) / latency,
            has_tool_call=True,
            correct_tool=correct_tool,
            correct_args=correct_args,
            correct_final_answer=False,
            tool_name=tool_name,
            tool_args=tool_args,
            final_text="",
            error=f"Tool execution failed: {exc}",
        )

    assistant_tool_message = {
        "role": "assistant",
        "content": first_msg.get("content") or "",
        "tool_calls": first_msg.get("tool_calls", []),
    }
    tool_message = {
        "role": "tool",
        "tool_call_id": tool_id or "call_1",
        "name": tool_name,
        "content": json.dumps(tool_result, ensure_ascii=False),
    }

    try:
        second_resp = post_chat_completion(
            base_url=deployment.base_url,
            model=deployment.model,
            messages=[user_message, assistant_tool_message, tool_message],
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            tools=tools,
            tool_choice="none",
        )
        t2 = time.time()
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, ValueError, OSError) as exc:
        t2 = time.time()
        total_latency = max(t2 - t0, 1e-9)
        total_tokens = max(1, first_tokens)
        return CaseResult(
            case_id=case.id,
            ok=False,
            first_call_latency_s=max(t1 - t0, 0.0),
            total_latency_s=total_latency,
            completion_tokens=total_tokens,
            token_speed_tps=total_tokens / total_latency,
            has_tool_call=True,
            correct_tool=correct_tool,
            correct_args=correct_args,
            correct_final_answer=False,
            tool_name=tool_name,
            tool_args=tool_args,
            final_text="",
            error=f"Second round failed: {exc}",
        )

    second_msg = get_message(second_resp)
    second_tokens = completion_tokens(second_resp)
    final_text = str(second_msg.get("content") or "")
    answer_ok = case.expected_answer_hint in final_text

    total_latency = max(t2 - t0, 1e-9)
    total_tokens = max(1, first_tokens + second_tokens)
    ok = has_tool_call and correct_tool and correct_args and answer_ok

    return CaseResult(
        case_id=case.id,
        ok=ok,
        first_call_latency_s=max(t1 - t0, 0.0),
        total_latency_s=total_latency,
        completion_tokens=total_tokens,
        token_speed_tps=total_tokens / total_latency,
        has_tool_call=has_tool_call,
        correct_tool=correct_tool,
        correct_args=correct_args,
        correct_final_answer=answer_ok,
        tool_name=tool_name,
        tool_args=tool_args,
        final_text=final_text,
        error=None,
    )


def summarize(results: list[CaseResult]) -> dict[str, float]:
    if not results:
        return {
            "overall_success_rate": 0.0,
            "tool_call_rate": 0.0,
            "correct_tool_rate": 0.0,
            "correct_args_rate": 0.0,
            "correct_final_answer_rate": 0.0,
            "avg_first_call_latency_s": 0.0,
            "avg_total_latency_s": 0.0,
            "avg_token_speed_tps": 0.0,
        }

    return {
        "overall_success_rate": statistics.mean(1.0 if r.ok else 0.0 for r in results),
        "tool_call_rate": statistics.mean(1.0 if r.has_tool_call else 0.0 for r in results),
        "correct_tool_rate": statistics.mean(1.0 if r.correct_tool else 0.0 for r in results),
        "correct_args_rate": statistics.mean(1.0 if r.correct_args else 0.0 for r in results),
        "correct_final_answer_rate": statistics.mean(1.0 if r.correct_final_answer else 0.0 for r in results),
        "avg_first_call_latency_s": statistics.mean(r.first_call_latency_s for r in results),
        "avg_total_latency_s": statistics.mean(r.total_latency_s for r in results),
        "avg_token_speed_tps": statistics.mean(r.token_speed_tps for r in results),
    }


def make_markdown_report(summary_by_dep: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Tool Calling Benchmark",
        "",
        "| Deployment | Overall Success | Tool Call Rate | Correct Tool | Correct Args | Correct Final Answer | Avg First Call Latency (s) | Avg Total Latency (s) | Avg Token/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summary_by_dep.items():
        lines.append(
            "| {name} | {overall_success_rate:.2%} | {tool_call_rate:.2%} | {correct_tool_rate:.2%} | "
            "{correct_args_rate:.2%} | {correct_final_answer_rate:.2%} | {avg_first_call_latency_s:.3f} | "
            "{avg_total_latency_s:.3f} | {avg_token_speed_tps:.2f} |".format(name=name, **s)
        )
    lines.append("")
    lines.append("Overall Success requires: tool call exists + correct tool + expected args + expected final answer.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark tool-calling capability on vLLM/llama.cpp deployments")
    p.add_argument(
        "--deployments",
        nargs="+",
        choices=["vllm_gpu", "vllm_cpu", "llamacpp_cpu"],
        default=["vllm_gpu", "vllm_cpu", "llamacpp_cpu"],
        help="Subset of deployments to benchmark",
    )
    p.add_argument("--requests-per-deployment", type=int, default=len(TOOL_CASES))
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--first-tool-choice",
        choices=["auto", "required"],
        default="required",
        help="tool_choice used in first round before executing tool",
    )
    p.add_argument("--output-dir", default="benchmark_results")

    p.add_argument("--vllm-gpu-url", default="http://127.0.0.1:8000")
    p.add_argument("--vllm-gpu-model", default="qwen3-1.7b")

    p.add_argument("--vllm-cpu-url", default="http://127.0.0.1:8001")
    p.add_argument("--vllm-cpu-model", default="qwen3-1.7b-cpu")

    p.add_argument("--llamacpp-url", default="http://127.0.0.1:8010")
    p.add_argument("--llamacpp-model", default="llama-cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    deployments_all = [
        Deployment("vllm_gpu", args.vllm_gpu_url, args.vllm_gpu_model),
        Deployment("vllm_cpu", args.vllm_cpu_url, args.vllm_cpu_model),
        Deployment("llamacpp_cpu", args.llamacpp_url, args.llamacpp_model),
    ]
    deployments = [d for d in deployments_all if d.name in set(args.deployments)]

    cases = TOOL_CASES[: max(1, min(args.requests_per_deployment, len(TOOL_CASES)))]

    all_results: dict[str, list[CaseResult]] = {d.name: [] for d in deployments}

    for dep in deployments:
        print(f"\n=== Benchmarking {dep.name} ({dep.base_url}) ===")

        # Warmup request is excluded from summary.
        try:
            _ = post_chat_completion(
                base_url=dep.base_url,
                model=dep.model,
                messages=[{"role": "user", "content": "请回复：答案: warmup"}],
                max_tokens=16,
                timeout_s=args.timeout,
            )
            print("Warmup: ok")
        except Exception as exc:  # noqa: BLE001
            print(f"Warmup failed: {exc}")

        for i, case in enumerate(cases, start=1):
            result = run_case(
                deployment=dep,
                case=case,
                timeout_s=args.timeout,
                max_tokens=args.max_tokens,
                first_tool_choice=args.first_tool_choice,
            )
            all_results[dep.name].append(result)
            status = "OK" if result.ok else "ERR"
            print(
                f"[{dep.name}] {i}/{len(cases)} {case.id} {status} "
                f"first={result.first_call_latency_s:.2f}s total={result.total_latency_s:.2f}s "
                f"tool={result.tool_name} args_ok={result.correct_args} ans_ok={result.correct_final_answer}"
            )

    summary_by_dep: dict[str, dict[str, float]] = {
        dep.name: summarize(all_results[dep.name]) for dep in deployments
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"toolcall_benchmark_{ts}.json"
    md_path = out_dir / f"toolcall_benchmark_{ts}.md"

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
