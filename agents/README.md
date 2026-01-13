# Agent Pipeline

This folder defines a 5-role workflow and a runner script that executes the
loop until the Manager marks the result as pass (unlimited rounds by default).

Roles
- planner: task breakdown and improvement plan
- implementer: code changes (must edit repo files)
- tester: run tests or add new tests
- reviewer: review diffs and risks
- manager: decide pass/fail and next steps

Round scope
- Each round targets 1-2 deliverables; if a round fails, the next Planner/Reviewer focus on fixing those issues before expanding scope.

Runner
```
python agents/run_pipeline.py --goal "<your request>"
```

Configuration
- `agents/pipeline.json` defines the agent order, pass values, and `max_rounds`.
- Set `max_rounds` to 0 (or less) for unlimited rounds; use `--max-rounds` to cap.
- The default `command_template` uses `python agents/run_codex_agent.py` as a wrapper.
- The wrapper normalizes agent output to JSON; invalid output is wrapped with an error payload.
- `capture_git` saves `git status` and `git diff` snapshots per round.
- `validate_outputs` enforces per-role JSON structure.
- `enforce_implementer_changes` fails the run if the Implementer makes no repo changes.
- Git snapshots are written to `agents/runs/<run_id>/round_<n>/git_*`.
- `command_template` can be set in `agents/pipeline.json` or via
  `AGENT_CMD_TEMPLATE` env var. The env var takes precedence.
- Template variables supported: `{agent_file}`, `{agent}`, `{input_file}`,
  `{output_file}`, `{round}`.

Example command template
```
export AGENT_CMD_TEMPLATE='python agents/run_codex_agent.py --agent-file {agent_file} --input {input_file} --output {output_file} --role {agent}'
```

Output format
Each agent must write a JSON object between markers:
```
===RESULT===
{ ... }
===END===
```

If no command template is set, the runner pauses and waits for you to fill each
output file manually.
