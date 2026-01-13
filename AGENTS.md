# Agents Workflow

This repo uses a 5-role loop to evolve changes:
Planner -> Implementer -> Tester -> Reviewer -> Manager -> decision.

Agent prompts live under `agents/` and the automation entrypoint is
`agents/run_pipeline.py`.

Quick start:
- Run the pipeline with the default wrapper (no env var needed).
- If no command template is set, the runner pauses for manual output.
- The default config runs without a round limit; use `--max-rounds` to cap.
- The default command template uses `python agents/run_codex_agent.py`; override it if needed.
- The wrapper normalizes agent output to JSON and warns on invalid output.

Example:
```
export AGENT_CMD_TEMPLATE='python agents/run_codex_agent.py --agent-file {agent_file} --input {input_file} --output {output_file} --role {agent}'
python agents/run_pipeline.py --goal "Improve Monte Carlo stability tests"
```

See `agents/README.md` for configuration details and output format.
