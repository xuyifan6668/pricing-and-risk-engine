"""Minimal orchestration helpers for agent command formatting."""

from __future__ import annotations

from dataclasses import dataclass
import shlex


@dataclass(frozen=True)
class AgentSpec:
    name: str
    path: str
    instructions: str


class CommandRunner:
    @staticmethod
    def _format_command_args(template: str, agent: AgentSpec) -> list[str]:
        return CommandRunner.format_command_args(template, agent)

    @staticmethod
    def format_command_args(template: str, agent: AgentSpec, **context: str) -> list[str]:
        params = {"agent_file": agent.path, "agent": agent.name, **context}
        formatted = template.format(**{key: shlex.quote(str(value)) for key, value in params.items()})
        return shlex.split(formatted)
