"""Wrapper to run Codex exec with agent instructions and input payload."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

RESULT_BLOCK = re.compile(r"===RESULT===\s*(\{.*?\})\s*===END===", re.S)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _format_payload(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()
    return json.dumps(data, indent=2, sort_keys=True)


def _build_prompt(agent_text: str, payload_text: str, role: str) -> str:
    agent_text = agent_text.strip()
    payload_text = payload_text.strip()
    return (
        f"{agent_text}\n\n"
        f"Role: {role}\n"
        "Input payload (JSON):\n"
        f"{payload_text}\n\n"
        "Follow the role instructions exactly.\n"
        "Return ONLY the JSON object between the markers; no extra text."
    )


def _find_json_object(text: str) -> str | None:
    last_valid = None
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                else:
                    last_valid = candidate
                start = None

    return last_valid


def _extract_payload(text: str) -> dict:
    match = RESULT_BLOCK.search(text)
    if match:
        return json.loads(match.group(1))

    stripped = text.strip()
    if stripped:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    candidate = _find_json_object(text)
    if candidate:
        return json.loads(candidate)

    raise ValueError("No JSON object found in output.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Codex agent with file inputs.")
    parser.add_argument("--agent-file", required=True, help="Path to the agent instructions file.")
    parser.add_argument("--input", required=True, help="Path to the input payload JSON.")
    parser.add_argument("--output", required=True, help="Path to write the agent output.")
    parser.add_argument("--role", required=True, help="Agent role name.")

    args = parser.parse_args()

    agent_path = Path(args.agent_file)
    input_path = Path(args.input)
    output_path = Path(args.output)

    agent_text = _read_text(agent_path)
    payload_text = _format_payload(_read_text(input_path))
    prompt = _build_prompt(agent_text, payload_text, args.role)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["codex", "exec", "--output-last-message", str(output_path), "-"]
    try:
        subprocess.run(cmd, input=prompt, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Codex exec failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode

    output_text = _read_text(output_path)
    try:
        payload = _extract_payload(output_text)
    except ValueError as exc:
        fallback = {
            "status": "fail",
            "error": str(exc),
            "raw_output": output_text.strip()[:2000],
        }
        print("Warning: agent output was not valid JSON; writing fallback payload.", file=sys.stderr)
        payload = fallback

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
