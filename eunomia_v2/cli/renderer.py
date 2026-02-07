"""Rich terminal renderer for pipeline stream events.

Provides real-time colored output during pipeline execution:
- Agent labels with distinct colors
- Task progress tracking (completed/total)
- QA pass/fail indicators
- Developer output summaries
"""

from typing import Any

from rich.console import Console

from eunomia_v2.graph.streaming import AGENT_DISPLAY, EventType, StreamEvent

# Agent color scheme — each agent gets a distinct color
AGENT_COLORS: dict[str, str] = {
    "planner": "blue",
    "task_router": "dim",
    "developer": "green",
    "qa": "yellow",
    "devops": "cyan",
    "architect": "magenta",
    "commit": "white",
    "escalate": "red",
}


class PipelineRenderer:
    """Renders pipeline stream events to the terminal using Rich.

    Tracks task progress and displays real-time agent activity
    with colored labels and progress indicators.
    """

    def __init__(self, console: Console | None = None, verbose: bool = False):
        self.console = console or Console()
        self.verbose = verbose
        self.total_tasks = 0
        self.completed_count = 0
        self.failed_count = 0
        self.active_agent = ""
        self.pending_interrupt: StreamEvent | None = None

    def render(self, event: StreamEvent) -> None:
        """Render a single stream event to the terminal."""
        handler = {
            EventType.PIPELINE_START.value: self._on_pipeline_start,
            EventType.PIPELINE_END.value: self._on_pipeline_end,
            EventType.AGENT_START.value: self._on_agent_start,
            EventType.AGENT_END.value: self._on_agent_end,
            EventType.TASK_START.value: self._on_task_start,
            EventType.TASK_COMPLETE.value: self._on_task_complete,
            EventType.TASK_FAILED.value: self._on_task_failed,
            EventType.STATE_UPDATE.value: self._on_state_update,
            EventType.TOKEN.value: self._on_token,
            EventType.TOOL_CALL.value: self._on_tool_call,
            EventType.ERROR.value: self._on_error,
            EventType.INTERRUPT.value: self._on_interrupt,
        }.get(event.type)

        if handler:
            handler(event)
        elif self.verbose:
            self.console.print(f"  [dim]{event.type}: {event.content}[/dim]")

    def _on_pipeline_start(self, event: StreamEvent) -> None:
        self.console.print()
        self.console.rule("[bold blue]Pipeline Started[/bold blue]")
        self.console.print()

    def _on_pipeline_end(self, event: StreamEvent) -> None:
        total = event.metadata.get("total_tasks", self.total_tasks)
        done = event.metadata.get("completed_count", self.completed_count)
        fail = event.metadata.get("failed_count", self.failed_count)
        skip = total - done - fail

        self.console.print()
        self.console.rule("[bold blue]Pipeline Complete[/bold blue]")
        self.console.print(
            f"  [green]{done} completed[/green]  "
            f"[red]{fail} failed[/red]  "
            f"[yellow]{skip} skipped[/yellow]  "
            f"[dim]{total} total[/dim]"
        )
        self.console.print()

    def _on_agent_start(self, event: StreamEvent) -> None:
        color = AGENT_COLORS.get(event.agent, "white")
        display = AGENT_DISPLAY.get(event.agent, event.agent)
        self.active_agent = event.agent

        # Progress indicator
        progress = ""
        if self.total_tasks > 0:
            progress = f" [{self.completed_count}/{self.total_tasks}]"

        self.console.print(
            f"  [{color}]>> {display}[/{color}]{progress}"
        )

    def _on_agent_end(self, event: StreamEvent) -> None:
        if self.verbose:
            color = AGENT_COLORS.get(event.agent, "white")
            display = AGENT_DISPLAY.get(event.agent, event.agent)
            self.console.print(f"  [{color}]<< {display} done[/{color}]")

    def _on_task_start(self, event: StreamEvent) -> None:
        task_id = event.metadata.get("task_id", "?")
        title = event.metadata.get("task_title", event.content)
        task_type = event.metadata.get("task_type", "")
        self.console.print(
            f"     [cyan][{task_id}][/cyan] [magenta]{task_type}[/magenta] {title}"
        )

    def _on_task_complete(self, event: StreamEvent) -> None:
        task_id = event.metadata.get("task_id", "?")
        self.completed_count = event.metadata.get("completed_count", self.completed_count + 1)
        self.total_tasks = event.metadata.get("total_tasks", self.total_tasks)
        self.console.print(
            f"     [green]Task [{task_id}] completed[/green] "
            f"[dim]({self.completed_count}/{self.total_tasks})[/dim]"
        )

    def _on_task_failed(self, event: StreamEvent) -> None:
        task_id = event.metadata.get("task_id", "?")
        self.failed_count = event.metadata.get("failed_count", self.failed_count + 1)
        self.console.print(f"     [red]Task [{task_id}] FAILED[/red]")

    def _on_state_update(self, event: StreamEvent) -> None:
        if "total_tasks" in event.metadata:
            self.total_tasks = event.metadata["total_tasks"]

        # Skip empty updates
        if not event.content:
            return

        color = AGENT_COLORS.get(event.agent, "white")

        # QA result — always show
        if "qa_passed" in event.metadata:
            passed = event.metadata["qa_passed"]
            result_color = "green" if passed else "red"
            self.console.print(f"     [{result_color}]{event.content}[/{result_color}]")
            return

        # Developer output — always show
        if "files_created" in event.metadata:
            self.console.print(f"     [{color}]{event.content}[/{color}]")
            return

        # Retry — always show
        if "retry_count" in event.metadata:
            self.console.print(f"     [yellow]{event.content}[/yellow]")
            return

        # Generated tasks — always show
        if "total_tasks" in event.metadata and "Generated" in event.content:
            self.console.print(f"     [{color}]{event.content}[/{color}]")
            return

        # Other updates — verbose only
        if self.verbose:
            self.console.print(f"     [dim]{event.content}[/dim]")

    def _on_token(self, event: StreamEvent) -> None:
        """Print LLM tokens inline (no newline)."""
        if self.verbose and event.content:
            self.console.print(event.content, end="")

    def _on_tool_call(self, event: StreamEvent) -> None:
        """Show tool invocations."""
        if self.verbose:
            tool_name = event.metadata.get("tool_name", "unknown")
            self.console.print(f"     [dim]tool: {tool_name}[/dim]")

    def _on_interrupt(self, event: StreamEvent) -> None:
        """Handle an interrupt event — store it and display a prompt."""
        self.pending_interrupt = event
        interrupt_type = event.metadata.get("interrupt_type", "unknown")
        agent = AGENT_DISPLAY.get(event.agent, event.agent) if event.agent else "Pipeline"

        self.console.print()
        self.console.rule("[bold yellow]Human Input Required[/bold yellow]")
        self.console.print(f"  [yellow]From:[/yellow] [bold]{agent}[/bold]")
        self.console.print(f"  [yellow]Type:[/yellow] {interrupt_type}")

        if event.content:
            self.console.print(f"  [yellow]Message:[/yellow] {event.content}")

        # Show task list for plan approval interrupts
        interrupt_data = event.metadata.get("interrupt_data", {})
        tasks = interrupt_data.get("tasks", [])
        if tasks:
            self.console.print(f"\n  [dim]Proposed tasks ({len(tasks)}):[/dim]")
            for i, t in enumerate(tasks, 1):
                title = t.get("title", t) if isinstance(t, dict) else str(t)
                self.console.print(f"    [dim]{i}.[/dim] {title}")

        # Show options for escalation interrupts
        options = interrupt_data.get("options", [])
        if options:
            self.console.print(f"\n  [dim]Options:[/dim] {', '.join(options)}")

        self.console.print()

    def _on_error(self, event: StreamEvent) -> None:
        exc_type = event.metadata.get("exception_type", "Error")
        self.console.print(f"  [bold red]{exc_type}: {event.content}[/bold red]")
