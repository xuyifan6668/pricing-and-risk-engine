"""Agent pipeline runner for the Planner->Implementer->Tester->Reviewer->Manager loop."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator import AgentSpec, CommandRunner

RESULT_BLOCK = re.compile(r"===RESULT===\s*(\{.*?\})\s*===END===", re.S)

SKELETONS: dict[str, dict[str, Any]] = {
    "planner": {
        "plan": [],
        "improvements": [],
        "acceptance_criteria": [],
        "risks": [],
        "questions": [],
    },
    "implementer": {
        "changes": [{"file": "path", "summary": ""}],
        "tests_needed": [],
        "notes": [],
        "open_questions": [],
    },
    "tester": {
        "tests_run": [{"command": "", "status": "passed|failed", "details": ""}],
        "new_tests": [],
        "failures": [],
        "notes": [],
    },
    "reviewer": {
        "findings": [{"severity": "high|medium|low", "file": "path", "detail": ""}],
        "approval": "approve|changes_requested",
        "notes": [],
    },
    "manager": {
        "status": "fail",
        "summary": "",
        "reasons": [],
        "next_steps": [],
    },
}

ROLE_SCHEMAS: dict[str, dict[str, type]] = {
    "planner": {
        "plan": list,
        "improvements": list,
        "acceptance_criteria": list,
        "risks": list,
        "questions": list,
    },
    "implementer": {
        "changes": list,
        "tests_needed": list,
        "notes": list,
        "open_questions": list,
    },
    "tester": {
        "tests_run": list,
        "new_tests": list,
        "failures": list,
        "notes": list,
    },
    "reviewer": {
        "findings": list,
        "approval": str,
        "notes": list,
    },
    "manager": {
        "status": str,
        "summary": str,
        "reasons": list,
        "next_steps": list,
    },
}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:40] if slug else "run"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _extract_payload(text: str) -> dict[str, Any]:
    match = RESULT_BLOCK.search(text)
    if match:
        payload_text = match.group(1)
    else:
        payload_text = text.strip()
    if not payload_text:
        return {}
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc


def _read_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing output file: {path}")
    text = path.read_text(encoding="utf-8")
    return _extract_payload(text)


def _resolve_template(config: dict[str, Any], explicit: str | None) -> str | None:
    template = explicit or os.getenv("AGENT_CMD_TEMPLATE") or config.get("command_template")
    if not template:
        return None
    template = template.strip()
    return template or None


def _run_git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_available(repo_root: Path) -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


def _capture_git_state(round_dir: Path, repo_root: Path, label: str) -> None:
    try:
        status = _run_git(repo_root, ["status", "-sb"])
        porcelain = _run_git(repo_root, ["status", "--porcelain", "-uall"])
        diff = _run_git(repo_root, ["diff"])
        diff_cached = _run_git(repo_root, ["diff", "--cached"])
        diff_stat = _run_git(repo_root, ["diff", "--stat"])
    except Exception as exc:
        print(f"Warning: failed to capture git state ({label}): {exc}", file=sys.stderr)
        return

    (round_dir / f"git_{label}_status.txt").write_text(status, encoding="utf-8")
    (round_dir / f"git_{label}_status_porcelain.txt").write_text(porcelain, encoding="utf-8")
    (round_dir / f"git_{label}_diff.patch").write_text(diff, encoding="utf-8")
    (round_dir / f"git_{label}_diff_cached.patch").write_text(diff_cached, encoding="utf-8")
    (round_dir / f"git_{label}_diff_stat.txt").write_text(diff_stat, encoding="utf-8")


def _validate_list_of_dicts(role: str, key: str, values: Any, required_keys: list[str]) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{role} output '{key}' must be a list.")
    for idx, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"{role} output '{key}[{idx}]' must be an object.")
        for required in required_keys:
            if required not in item:
                raise ValueError(f"{role} output '{key}[{idx}]' missing '{required}'.")


def _validate_payload(role: str, payload: dict[str, Any]) -> None:
    schema = ROLE_SCHEMAS.get(role)
    if not schema:
        return
    for key, expected in schema.items():
        if key not in payload:
            raise ValueError(f"{role} output missing required key '{key}'.")
        if not isinstance(payload[key], expected):
            raise ValueError(f"{role} output '{key}' must be {expected.__name__}.")

    if role == "implementer":
        _validate_list_of_dicts(role, "changes", payload["changes"], ["file", "summary"])
    elif role == "tester":
        _validate_list_of_dicts(role, "tests_run", payload["tests_run"], ["command", "status", "details"])
    elif role == "reviewer":
        _validate_list_of_dicts(role, "findings", payload["findings"], ["severity", "file", "detail"])


def _build_agent_specs(config: dict[str, Any], repo_root: Path) -> list[AgentSpec]:
    agents = []
    for entry in config.get("agents", []):
        path = repo_root / entry["file"]
        instructions = path.read_text(encoding="utf-8")
        agents.append(AgentSpec(name=entry["name"], path=str(path), instructions=instructions))
    return agents


def _write_input(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _write_output_skeleton(path: Path, agent_name: str) -> None:
    skeleton = SKELETONS.get(agent_name, {})
    content = "===RESULT===\n" + json.dumps(skeleton, indent=2, sort_keys=True) + "\n===END===\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_agent_command(
    template: str,
    agent: AgentSpec,
    input_path: Path,
    output_path: Path,
    round_idx: int,
    repo_root: Path,
) -> None:
    args = CommandRunner.format_command_args(
        template,
        agent,
        input_file=str(input_path),
        output_file=str(output_path),
        round=round_idx,
    )
    subprocess.run(args, check=True, cwd=repo_root)


def _manual_wait(output_path: Path) -> None:
    prompt = f"Fill {output_path} then press Enter to continue..."
    input(prompt)


def _manager_passed(payload: dict[str, Any], status_keys: list[str], pass_values: list[str]) -> bool:
    normalized_pass = {value.strip().lower() for value in pass_values}
    for key in status_keys:
        if key in payload:
            value = str(payload[key]).strip().lower()
            return value in normalized_pass
    return False


def run_pipeline(goal: str, config_path: Path, max_rounds: int | None, manual: bool, template: str | None) -> int:
    repo_root = REPO_ROOT
    config = _load_json(config_path)
    if max_rounds is not None:
        config["max_rounds"] = max_rounds

    agents = _build_agent_specs(config, repo_root)
    if not agents:
        raise ValueError("No agents configured.")

    template = _resolve_template(config, template)
    manual = manual or template is None

    run_dir = repo_root / "agents" / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{_slugify(goal)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_state: dict[str, Any] = {"goal": goal, "rounds": []}

    max_rounds_raw = config.get("max_rounds", 1)
    max_rounds = int(max_rounds_raw) if max_rounds_raw is not None else None
    capture_git = bool(config.get("capture_git", True))
    validate_outputs = bool(config.get("validate_outputs", True))
    enforce_implementer_changes = bool(config.get("enforce_implementer_changes", True))
    git_enabled = capture_git or enforce_implementer_changes
    git_available = _git_available(repo_root) if git_enabled else False
    status_keys = list(config.get("status_keys", ["status"]))
    pass_values = list(config.get("pass_values", ["pass"]))

    unlimited = max_rounds is None or max_rounds <= 0
    round_idx = 1
    while unlimited or round_idx <= max_rounds:
        round_dir = run_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        if capture_git and git_available:
            _capture_git_state(round_dir, repo_root, "pre")

        round_outputs: dict[str, Any] = {}
        past_rounds_summary = [
            {"round": past["round"], "manager": past.get("outputs", {}).get("manager")}
            for past in run_state["rounds"]
        ]

        for agent in agents:
            pre_implementer_status = None
            if (
                agent.name == "implementer"
                and enforce_implementer_changes
                and git_available
            ):
                try:
                    pre_implementer_status = _run_git(repo_root, ["status", "--porcelain", "-uall"])
                except Exception as exc:
                    print(f"Warning: failed to read git status: {exc}", file=sys.stderr)
                    pre_implementer_status = None

            input_path = round_dir / f"{agent.name}_input.json"
            output_path = round_dir / f"{agent.name}_output.json"

            input_payload = {
                "goal": goal,
                "round": round_idx,
                "role": agent.name,
                "instructions": agent.instructions,
                "repo_root": str(repo_root),
                "run_dir": str(run_dir),
                "previous_outputs": round_outputs,
                "past_rounds": past_rounds_summary,
            }
            _write_input(input_path, input_payload)

            if manual:
                _write_output_skeleton(output_path, agent.name)
                _manual_wait(output_path)
            else:
                _run_agent_command(template, agent, input_path, output_path, round_idx, repo_root)

            output_payload = _read_output(output_path)
            if validate_outputs:
                _validate_payload(agent.name, output_payload)
            round_outputs[agent.name] = output_payload

            if (
                agent.name == "implementer"
                and enforce_implementer_changes
                and git_available
                and pre_implementer_status is not None
            ):
                try:
                    post_status = _run_git(repo_root, ["status", "--porcelain", "-uall"])
                except Exception as exc:
                    print(f"Warning: failed to read git status: {exc}", file=sys.stderr)
                else:
                    if post_status == pre_implementer_status:
                        raise ValueError(
                            "Implementer made no repo changes; edit files or disable enforce_implementer_changes."
                        )

        run_state["rounds"].append({"round": round_idx, "outputs": round_outputs})
        _write_json(run_dir / "summary.json", run_state)

        if capture_git and git_available:
            _capture_git_state(round_dir, repo_root, "post")

        manager_payload = round_outputs.get("manager", {})
        if _manager_passed(manager_payload, status_keys, pass_values):
            return 0

        round_idx += 1

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multi-agent pipeline.")
    parser.add_argument("--goal", type=str, help="Goal input for the pipeline.")
    parser.add_argument("--goal-file", type=str, help="Path to a file containing the goal text.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config JSON.")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Override max rounds (0 or less means unlimited).",
    )
    parser.add_argument("--manual", action="store_true", help="Force manual output mode.")
    parser.add_argument("--command-template", type=str, default=None, help="Override command template.")

    args = parser.parse_args()

    if args.goal_file:
        goal = Path(args.goal_file).read_text(encoding="utf-8").strip()
    else:
        goal = (args.goal or "").strip()

    if not goal:
        print("Goal is required via --goal or --goal-file.", file=sys.stderr)
        return 2

    repo_root = REPO_ROOT
    config_path = Path(args.config) if args.config else repo_root / "agents" / "pipeline.json"

    try:
        return run_pipeline(goal, config_path, args.max_rounds, args.manual, args.command_template)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
