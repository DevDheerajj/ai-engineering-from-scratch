import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
PROFILES = {
    "gb10": {"model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "fp8": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "draft": "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex", "fraction": "0.7"},
    "l4": {"model": "Qwen/Qwen3-8B", "fp8": "Qwen/Qwen3-8B-FP8", "draft": "Tengyunw/qwen3_8b_eagle3", "fraction": "0.85"},
}
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64]
RATES = [0.5, 1, 2, 4, 8, 16]
PREFIXES = [0, 256, 1024, 2048, 4096]
SEEDS = [1, 2, 3]
RANDOM = ["--dataset-name", "random", "--random-input-len", "512", "--random-output-len", "128", "--random-range-ratio", "1.0"]


def configs(profile):
    p = PROFILES[profile]
    eagle = ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", p["draft"], "--speculative-num-steps", "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]
    return {
        "default": (p["model"], []),
        "fp8": (p["fp8"], ["--kv-cache-dtype", "fp8_e4m3"]),
        "eagle3": (p["model"], eagle, {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}),
        "no-cuda-graph": (p["model"], ["--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled"]),
        "no-radix-cache": (p["model"], ["--disable-radix-cache"]),
    }


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def start_server(model, flags, port, fraction, log_path, env=None):
    if healthy(port):
        raise RuntimeError(f"another server already answers on port {port}")
    command = [shutil.which("sglang") or "sglang", "serve", "--model-path", model, "--host", "127.0.0.1", "--port", str(port), "--enable-cache-report", "--enable-metrics", "--mem-fraction-static", fraction, *flags]
    process = subprocess.Popen(command, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})}, start_new_session=True)
    deadline = time.time() + 1800
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}, see {log_path}")
        if healthy(port):
            return process
        time.sleep(2)
    stop_server(process, port)
    raise RuntimeError(f"server did not become healthy, see {log_path}")


def stop_server(process, port):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + 60
    while healthy(port) and time.time() < deadline:
        time.sleep(1)


def bench(port, output, args):
    command = [PYTHON, "-m", "sglang.benchmark.serving", "--backend", "sglang", "--port", str(port), "--output-file", str(output), "--output-details", *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=7200, stdin=subprocess.DEVNULL)
    output.with_suffix(".txt").write_text((result.stdout + result.stderr).replace(str(Path.home()), "~"))
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed, see {output.with_suffix('.txt')}")


def gpu_index():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def temperature(index):
    result = subprocess.run(["nvidia-smi", "-i", index, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def wait_cool(index, limit, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        current = temperature(index)
        if current is None or current <= limit:
            return current, round(time.time() - start)
        time.sleep(5)
    return temperature(index), round(time.time() - start)


def telemetry(index, path):
    query = "timestamp,temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.active,utilization.gpu"
    return subprocess.Popen(["nvidia-smi", "-i", index, f"--query-gpu={query}", "--format=csv,noheader,nounits", "-lms", "1000"], stdout=open(path, "w"), stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def plan(sweep, config):
    runs = []
    if sweep == "load":
        for seed in SEEDS:
            for c in CONCURRENCY:
                runs.append((f"c{c}-s{seed}", [*RANDOM, "--seed", str(seed), "--max-concurrency", str(c), "--num-prompts", str(max(16, 4 * c))]))
    elif sweep == "rate":
        for seed in SEEDS:
            for rate in RATES:
                runs.append((f"r{rate}-s{seed}", [*RANDOM, "--seed", str(seed), "--request-rate", str(rate), "--num-prompts", str(max(32, int(rate * 30)))]))
    elif sweep == "waves":
        for seed in SEEDS:
            for ratio in (1.0, 0.5, 0.0):
                runs.append((f"w{ratio}-s{seed}", ["--dataset-name", "random", "--random-input-len", "512", "--random-output-len", "128", "--random-range-ratio", str(ratio), "--seed", str(seed), "--max-concurrency", "16", "--num-prompts", "64"]))
    elif sweep == "prefix":
        for seed in SEEDS:
            for prefix in PREFIXES:
                runs.append((f"p{prefix}-s{seed}", ["--dataset-name", "generated-shared-prefix", "--gsp-num-groups", "4", "--gsp-prompts-per-group", "16", "--gsp-system-prompt-len", str(max(prefix, 1)), "--gsp-question-len", "128", "--gsp-output-len", "64", "--gsp-range-ratio", "1.0", "--max-concurrency", "8", "--seed", str(seed)]))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--sweep", required=True, choices=["load", "rate", "prefix", "waves"])
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--fraction", help="override the memory fraction of the profile")
    parser.add_argument("--cool", type=int, default=65, help="wait until the GPU is at or below this temperature before each run")
    args = parser.parse_args()
    table = configs(args.profile)
    for name in args.configs:
        model, flags, *rest = table[name]
        env = rest[0] if rest else {}
        directory = HERE / "runs" / args.profile / args.sweep / name
        directory.mkdir(parents=True, exist_ok=True)
        pending = [(run, bench_args) for run, bench_args in plan(args.sweep, name) if not (directory / f"{run}.jsonl").exists()]
        if not pending:
            print(f"[{name}] complete", flush=True)
            continue
        process = start_server(model, flags, args.port, args.fraction or PROFILES[args.profile]["fraction"], directory / "server.log", env)
        (directory / "server.command.txt").write_text(" ".join([*(f"{k}={v}" for k, v in env.items()), "sglang", *process.args[1:]]) + "\n")
        try:
            for run, bench_args in pending:
                start_temp, waited = wait_cool(gpu_index(), args.cool)
                print(f"[{name}] {run} start {start_temp} C after {waited} s", flush=True)
                sampler = telemetry(gpu_index(), directory / f"{run}.gpu.csv")
                try:
                    bench(args.port, directory / f"{run}.jsonl", bench_args)
                finally:
                    sampler.terminate()
                    sampler.wait()
                if process.poll() is not None:
                    raise RuntimeError(f"[{name}] server exited during {run}")
        finally:
            stop_server(process, args.port)
    print("done", flush=True)


if __name__ == "__main__":
    main()
