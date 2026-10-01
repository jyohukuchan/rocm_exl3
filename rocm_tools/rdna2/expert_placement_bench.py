"""Cold chat requests for static expert-placement comparisons (no profiling)."""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="JSON fixture with train/eval lists")
    parser.add_argument("--split", choices=("train", "eval"), default="eval")
    parser.add_argument("--base-url", default="http://127.0.0.1:3953")
    parser.add_argument("--model", default="qwen38-local")
    parser.add_argument("--key-env", default="EXL3_API_KEY")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--start-repeat", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup", action="store_true", help="Use before eval, never before route capture")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.max_tokens < 1:
        parser.error("--repeat and --max-tokens must be positive")
    if args.warmup and args.split == "train":
        parser.error("Warmup changes profile case numbering; omit it during capture")
    fixture = json.loads(Path(args.cases).read_text())
    key = os.environ.get(args.key_env)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key

    def request(prompt, max_tokens):
        payload = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0, "stream": False,
                   "enable_thinking": True, "reasoning_effort": "xhigh", "include_timings": False}
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=360) as response:
            return json.load(response)

    report = {"mode": args.split, "model": args.model, "max_tokens": args.max_tokens, "requests": []}
    if args.warmup:
        report["warmup"] = request("Warm up: briefly explain a binary search in Japanese.", 128)
    for repeat in range(args.start_repeat, args.start_repeat + args.repeat):
        for case in fixture[args.split]:
            prompt = f"Document nonce {repeat} {case['id']}.\n" + case["prompt"]
            start = time.monotonic()
            response = request(prompt, args.max_tokens)
            cached = response["usage"]["prompt_tokens_details"]["cached_tokens"]
            if cached:
                raise RuntimeError("Cold benchmark reused cache; restart or use a new --start-repeat")
            message = response["choices"][0]["message"]
            if not (message.get("content") or message.get("reasoning_content")):
                raise RuntimeError("Empty model response")
            report["requests"].append({"case": case["id"], "category": case["category"],
                                       "repeat": repeat, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                       "wall_seconds": time.monotonic() - start, "response": response})
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(repeat, case["id"], response["exl3_metrics"]["output_tokens_per_second"], flush=True)
    req = urllib.request.Request(args.base_url.rstrip("/") + "/props", headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        report["props"] = json.load(response)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
