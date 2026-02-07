"""Eunomia V2 CLI — main entry point.

Commands:
    plan   — Generate a task graph from a PRD file
    run    — Execute the full pipeline (plan → code → test → commit)
    status — Show pipeline results from last run
    serve  — Start the FastAPI server
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="eunomia-v2",
    help="Autonomous multi-agent software development framework",
    add_completion=False,
)
console = Console()

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"
SESSION_DIR = ".eunomia"
SESSION_FILE = "session.json"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _load_env() -> None:
    """Load .env from CWD or parent directories."""
    from dotenv import load_dotenv

    for search_dir in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent.parent]:
        env_file = search_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return
    load_dotenv()


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging — verbose shows agent-level detail."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    # Reduce noise
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _save_session(project_path: Path, session_data: dict[str, Any]) -> Path:
    """Save session state to .eunomia/session.json."""
    session_dir = project_path / SESSION_DIR
    session_dir.mkdir(parents=True, exist_ok=True)
    session_file = session_dir / SESSION_FILE
    session_file.write_text(
        json.dumps(session_data, indent=2, default=str),
        encoding="utf-8",
    )
    return session_file


def _load_session(project_path: Path) -> dict[str, Any] | None:
    """Load session state from .eunomia/session.json."""
    session_file = project_path / SESSION_DIR / SESSION_FILE
    if not session_file.exists():
        return None
    try:
        return json.loads(session_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _serialize_task(task: Any) -> dict[str, Any]:
    """Serialize a Task object to a JSON-safe dict."""
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description,
        "task_type": task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type),
        "filename": task.filename,
        "dependencies": list(task.dependencies),
        "acceptance_criteria": list(task.acceptance_criteria),
        "status": task.status.value if hasattr(task.status, "value") else str(task.status),
    }


async def _close_checkpointer(checkpointer: Any) -> None:
    """Close the checkpointer's database connection to prevent process hang.

    AsyncSqliteSaver holds an aiosqlite connection with a background thread.
    If not explicitly closed, the thread keeps the Python process alive after
    asyncio.run() completes.
    """
    conn = getattr(checkpointer, "conn", None)
    if conn is not None and hasattr(conn, "close"):
        try:
            await conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------
# Pipeline execution helpers (streaming + batch)
# ---------------------------------------------------------------

def _run_pipeline_batch(
    project_dir: Path,
    prd_content: str,
    model: str,
    max_retries: int,
    session_id: str,
    verbose: bool,
    hitl_level: str = "autonomous",
    checkpointer_backend: str = "memory",
) -> dict[str, Any]:
    """Run the pipeline with ainvoke() — no real-time output."""

    async def _invoke() -> dict[str, Any]:
        from eunomia_v2.graph.orchestrator import compile_graph
        from eunomia_v2.graph.state import create_initial_state
        from eunomia_v2.persistence.checkpointer import get_checkpointer

        state = create_initial_state(
            project_path=str(project_dir),
            prd_content=prd_content,
            model=model,
            max_retries=max_retries,
            session_id=session_id,
            hitl_level=hitl_level,
        )
        checkpointer = get_checkpointer(checkpointer_backend)
        try:
            graph = compile_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": f"cli-{session_id}"}}
            console.print("\n[bold]Starting pipeline (batch mode)...[/bold]\n")
            return await graph.ainvoke(state, config=config)
        finally:
            await _close_checkpointer(checkpointer)

    try:
        return asyncio.run(_invoke())
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


def _run_pipeline_streaming(
    project_dir: Path,
    prd_content: str,
    model: str,
    max_retries: int,
    session_id: str,
    verbose: bool,
    hitl_level: str = "autonomous",
    checkpointer_backend: str = "memory",
) -> dict[str, Any]:
    """Run the pipeline with astream() — real-time Rich output.

    When hitl_level is not autonomous, enters an interrupt/resume loop:
    after the initial stream, checks for pending interrupts, prompts the
    user for input, then resumes the pipeline with their response.
    """

    async def _stream() -> dict[str, Any]:
        from eunomia_v2.cli.renderer import PipelineRenderer
        from eunomia_v2.graph.orchestrator import compile_graph
        from eunomia_v2.graph.state import create_initial_state
        from eunomia_v2.graph.streaming import resume_pipeline, stream_pipeline
        from eunomia_v2.persistence.checkpointer import get_checkpointer

        state = create_initial_state(
            project_path=str(project_dir),
            prd_content=prd_content,
            model=model,
            max_retries=max_retries,
            session_id=session_id,
            hitl_level=hitl_level,
        )
        checkpointer = get_checkpointer(checkpointer_backend)
        graph = compile_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": f"cli-{session_id}"}}

        renderer = PipelineRenderer(console=console, verbose=verbose)

        try:
            # Initial stream
            async for event in stream_pipeline(graph, state, config):
                renderer.render(event)

            # Interrupt/resume loop — only when HITL is active
            while renderer.pending_interrupt is not None:
                interrupt_event = renderer.pending_interrupt
                interrupt_data = interrupt_event.metadata.get("interrupt_data", {})
                interrupt_type = interrupt_data.get("type", "unknown")

                # Prompt user for input based on interrupt type
                if interrupt_type in ("plan_approval", "task_approval", "commit_approval"):
                    user_input = console.input(
                        "[bold yellow]> Approve? (yes/no/feedback): [/bold yellow]"
                    )
                elif interrupt_type == "escalation":
                    user_input = console.input(
                        "[bold yellow]> Action (retry/skip/instructions): [/bold yellow]"
                    )
                else:
                    user_input = console.input(
                        "[bold yellow]> Your response: [/bold yellow]"
                    )

                # Clear pending interrupt before resuming
                renderer.pending_interrupt = None

                # Resume the pipeline with user input
                async for event in resume_pipeline(graph, user_input, config):
                    renderer.render(event)

            # Retrieve final accumulated state from the checkpointer
            snapshot = await graph.aget_state(config)
            return snapshot.values
        finally:
            await _close_checkpointer(checkpointer)

    try:
        return asyncio.run(_stream())
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


# ---------------------------------------------------------------
# plan command
# ---------------------------------------------------------------

@app.command()
def plan(
    prd: str = typer.Option(..., "--prd", help="Path to PRD file"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="LLM model to use"),
    output: str = typer.Option(
        "tasks.json", "--output", "-o", help="Output path for generated task graph",
    ),
) -> None:
    """Generate a task graph from a PRD file."""
    _load_env()
    _setup_logging()

    prd_path = Path(prd)
    if not prd_path.exists():
        console.print(f"[red]Error: PRD file not found: {prd_path}[/red]")
        raise typer.Exit(1)

    prd_content = prd_path.read_text(encoding="utf-8")

    console.print(Panel(
        f"[bold]PRD:[/bold] {prd_path.name}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Output:[/bold] {output}",
        title="[bold blue]Eunomia V2 — Plan[/bold blue]",
    ))

    # Run the planner
    async def _plan() -> Any:
        from eunomia_v2.agents.planner import generate_task_graph
        return await generate_task_graph(prd_content, model=model)

    console.print("\nGenerating task graph...\n")

    try:
        task_graph = asyncio.run(_plan())
    except Exception as e:
        console.print(f"[red]Planner failed: {e}[/red]")
        raise typer.Exit(1)

    # Display tasks
    table = Table(title="Generated Tasks")
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Type", style="magenta", width=16)
    table.add_column("Title", style="white")
    table.add_column("Deps", style="dim", width=8)

    for task in task_graph.tasks:
        deps_str = ",".join(str(d) for d in task.dependencies) if task.dependencies else "-"
        table.add_row(
            str(task.id),
            task.task_type.value,
            task.title,
            deps_str,
        )

    console.print(table)

    # Save to file
    output_path = Path(output)
    tasks_data = {
        "project_name": task_graph.project_name,
        "prd_summary": task_graph.prd_summary,
        "tasks": [_serialize_task(t) for t in task_graph.tasks],
    }
    output_path.write_text(
        json.dumps(tasks_data, indent=2),
        encoding="utf-8",
    )

    console.print(f"\n[green]Task graph saved to {output_path} ({len(task_graph.tasks)} tasks)[/green]")


# ---------------------------------------------------------------
# run command
# ---------------------------------------------------------------

@app.command()
def run(
    project_path: str = typer.Option(..., "--project-path", help="Project working directory"),
    prd: str = typer.Option(None, "--prd", help="Path to PRD file"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="LLM model to use"),
    max_retries: int = typer.Option(3, "--max-retries", help="Max QA-Dev feedback loop retries"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable real-time streaming output"),
    hitl: str = typer.Option(
        "autonomous", "--hitl",
        help="Human-in-the-loop level: autonomous, approval, or interactive",
    ),
    persist: bool = typer.Option(True, "--persist/--no-persist", help="Enable SQLite persistence"),
    ci: bool = typer.Option(False, "--ci", help="CI mode: no color, no prompts, JSON output, exit codes"),
) -> None:
    """Execute the full pipeline — plan, code, test, commit."""
    _load_env()
    _setup_logging(verbose)

    # CI mode overrides: no color, autonomous, batch mode
    if ci:
        no_stream = True
        hitl = "autonomous"
        console._force_terminal = False  # noqa: SLF001

    # Fix Windows console encoding
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    project_dir = Path(project_path).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # Resolve PRD content
    prd_content = ""
    if prd:
        prd_path = Path(prd)
        if not prd_path.exists():
            console.print(f"[red]Error: PRD file not found: {prd_path}[/red]")
            raise typer.Exit(1)
        prd_content = prd_path.read_text(encoding="utf-8")
    else:
        # Check for PRD in project directory
        for candidate in ["PRD.md", "prd.md", "README.md"]:
            candidate_path = project_dir / candidate
            if candidate_path.exists():
                prd_content = candidate_path.read_text(encoding="utf-8")
                console.print(f"[dim]Using PRD from {candidate_path}[/dim]")
                break

    if not prd_content:
        console.print("[red]Error: No PRD file provided. Use --prd or place PRD.md in project directory.[/red]")
        raise typer.Exit(1)

    # Validate hitl level
    valid_hitl = {"autonomous", "approval", "interactive"}
    if hitl not in valid_hitl:
        console.print(f"[red]Error: --hitl must be one of: {', '.join(sorted(valid_hitl))}[/red]")
        raise typer.Exit(1)

    # Non-autonomous HITL requires streaming for the interrupt/resume loop
    if no_stream and hitl != "autonomous":
        console.print(
            f"[yellow]Warning: --hitl={hitl} requires streaming. "
            f"Ignoring --no-stream.[/yellow]"
        )
        no_stream = False

    session_id = str(uuid.uuid4())[:8]

    # Persistence: save session metadata and use SQLite checkpointer
    session_store = None
    if persist:
        from eunomia_v2.persistence.session_store import SQLiteSessionStore
        session_store = SQLiteSessionStore()
        session_store.save_session(
            session_id=session_id,
            status="running",
            model=model,
            project_path=str(project_dir),
            prd_content=prd_content[:5000],
            max_retries=max_retries,
            hitl_level=hitl,
        )

    console.print(Panel(
        f"[bold]Project:[/bold] {project_dir}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Max retries:[/bold] {max_retries}\n"
        f"[bold]HITL:[/bold] {hitl}\n"
        f"[bold]Persist:[/bold] {'sqlite' if persist else 'memory'}\n"
        f"[bold]Session:[/bold] {session_id}",
        title="[bold blue]Eunomia V2 — Run Pipeline[/bold blue]",
    ))

    checkpointer_backend = "sqlite" if persist else "memory"

    if no_stream:
        # Non-streaming mode: use ainvoke() — shows only final result
        final_state = _run_pipeline_batch(
            project_dir, prd_content, model, max_retries, session_id, verbose,
            hitl_level=hitl, checkpointer_backend=checkpointer_backend,
        )
    else:
        # Streaming mode (default): real-time Rich output
        final_state = _run_pipeline_streaming(
            project_dir, prd_content, model, max_retries, session_id, verbose,
            hitl_level=hitl, checkpointer_backend=checkpointer_backend,
        )

    # Extract results
    tasks = final_state.get("tasks", [])
    completed = set(final_state.get("completed_tasks", []))
    failed = set(final_state.get("failed_tasks", []))

    total = len(tasks)
    n_done = len(completed)
    n_fail = len(failed)
    n_skip = total - n_done - n_fail

    if ci:
        # CI mode: JSON output to stdout
        ci_result = {
            "session_id": session_id,
            "status": "success" if n_fail == 0 else "failure",
            "total": total,
            "completed": n_done,
            "failed": n_fail,
            "skipped": n_skip,
            "tasks": [
                {
                    "id": str(t.id),
                    "title": t.title,
                    "type": t.task_type.value,
                    "status": "completed" if str(t.id) in completed
                    else "failed" if str(t.id) in failed
                    else "skipped",
                }
                for t in tasks
            ],
        }
        print(json.dumps(ci_result, indent=2))
    else:
        # Interactive mode: Rich tables
        table = Table(title="Pipeline Results")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Status", width=6)
        table.add_column("Type", style="magenta", width=16)
        table.add_column("Title", style="white")

        for task in tasks:
            tid = str(task.id)
            if tid in completed:
                status = "[green]DONE[/green]"
            elif tid in failed:
                status = "[red]FAIL[/red]"
            else:
                status = "[yellow]SKIP[/yellow]"
            table.add_row(
                str(task.id),
                status,
                task.task_type.value,
                task.title,
            )

        console.print(table)

        if n_fail == 0 and n_done == total:
            summary_color = "green"
            summary_emoji = "All tasks completed successfully."
        elif n_fail > 0:
            summary_color = "red"
            summary_emoji = f"{n_fail} task(s) failed. Check logs for details."
        else:
            summary_color = "yellow"
            summary_emoji = f"{n_done}/{total} completed, {n_skip} skipped."

        console.print(Panel(
            f"[bold]Total:[/bold] {total}  |  "
            f"[green]Completed:[/green] {n_done}  |  "
            f"[red]Failed:[/red] {n_fail}  |  "
            f"[yellow]Skipped:[/yellow] {n_skip}\n\n"
            f"[{summary_color}]{summary_emoji}[/{summary_color}]",
            title=f"[bold blue]Summary — Session {session_id}[/bold blue]",
        ))

        # List created files
        if project_dir.exists():
            skip_dirs = {".git", "node_modules", "__pycache__", ".venv", ".eunomia"}
            project_files = []
            for item in sorted(project_dir.rglob("*")):
                if item.is_file():
                    parts = item.relative_to(project_dir).parts
                    if any(p in skip_dirs for p in parts):
                        continue
                    project_files.append(str(item.relative_to(project_dir)))

            if project_files:
                console.print(f"\n[bold]Files in project ({len(project_files)}):[/bold]")
                for f in project_files:
                    console.print(f"  {f}")

    # Save session (local JSON)
    session_data = {
        "session_id": session_id,
        "model": model,
        "project_path": str(project_dir),
        "max_retries": max_retries,
        "total_tasks": total,
        "completed": list(completed),
        "failed": list(failed),
        "tasks": [_serialize_task(t) for t in tasks],
    }
    session_file = _save_session(project_dir, session_data)
    if not ci:
        console.print(f"\n[dim]Session saved to {session_file}[/dim]")

    # Persist final results to SQLite
    if session_store is not None:
        final_status = "completed" if n_fail == 0 else "failed"
        session_store.update_session(
            session_id,
            status=final_status,
            total_tasks=total,
            completed_tasks=list(completed),
            failed_tasks=list(failed),
        )

    # CI mode: exit code 1 on any failure
    if ci and n_fail > 0:
        raise typer.Exit(1)


# ---------------------------------------------------------------
# status command
# ---------------------------------------------------------------

@app.command()
def status(
    project_path: str = typer.Option(".", "--project-path", help="Project directory to check"),
) -> None:
    """Show the pipeline status from the last run."""
    project_dir = Path(project_path).resolve()

    session_data = _load_session(project_dir)
    if not session_data:
        console.print(f"[yellow]No session found in {project_dir / SESSION_DIR}[/yellow]")
        console.print("Run [bold]eunomia-v2 run --project-path ... --prd ...[/bold] first.")
        raise typer.Exit(1)

    session_id = session_data.get("session_id", "unknown")
    tasks = session_data.get("tasks", [])
    completed = set(session_data.get("completed", []))
    failed = set(session_data.get("failed", []))

    # Display results table
    table = Table(title=f"Session {session_id}")
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Status", width=6)
    table.add_column("Type", style="magenta", width=16)
    table.add_column("Title", style="white")

    for task in tasks:
        tid = str(task.get("id", ""))
        if tid in completed:
            task_status = "[green]DONE[/green]"
        elif tid in failed:
            task_status = "[red]FAIL[/red]"
        else:
            task_status = "[yellow]SKIP[/yellow]"
        table.add_row(
            tid,
            task_status,
            task.get("task_type", ""),
            task.get("title", ""),
        )

    console.print(table)

    total = len(tasks)
    n_done = len(completed)
    n_fail = len(failed)

    console.print(Panel(
        f"[bold]Model:[/bold] {session_data.get('model', 'unknown')}\n"
        f"[bold]Project:[/bold] {session_data.get('project_path', 'unknown')}\n"
        f"[bold]Total:[/bold] {total}  |  "
        f"[green]Completed:[/green] {n_done}  |  "
        f"[red]Failed:[/red] {n_fail}  |  "
        f"[yellow]Skipped:[/yellow] {total - n_done - n_fail}",
        title=f"[bold blue]Session {session_id}[/bold blue]",
    ))


# ---------------------------------------------------------------
# chat command — interactive PRD editor
# ---------------------------------------------------------------

@app.command()
def chat(
    project_path: str = typer.Option(".", "--project-path", help="Project working directory"),
    prd: str = typer.Option(None, "--prd", help="Path to existing PRD file to load"),
) -> None:
    """Interactive PRD chat — draft, refine, then run the pipeline."""
    project_dir = Path(project_path).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    prd_content = ""
    if prd:
        prd_path = Path(prd)
        if prd_path.exists():
            prd_content = prd_path.read_text(encoding="utf-8")
            console.print(f"[dim]Loaded PRD from {prd_path} ({len(prd_content)} chars)[/dim]")

    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  [cyan]show[/cyan]       — Display current PRD\n"
        "  [cyan]append[/cyan]     — Add text to the PRD\n"
        "  [cyan]replace[/cyan]    — Replace entire PRD content\n"
        "  [cyan]save[/cyan]       — Save PRD to file\n"
        "  [cyan]plan[/cyan]       — Generate task graph from current PRD\n"
        "  [cyan]run[/cyan]        — Execute the full pipeline\n"
        "  [cyan]generate[/cyan]   — Generate a document (prd, tdd, adr, readme, test_plan, runbook)\n"
        "  [cyan]quit[/cyan]       — Exit chat",
        title="[bold blue]Eunomia V2 — Chat[/bold blue]",
    ))

    while True:
        try:
            cmd = console.input("\n[bold cyan]chat>[/bold cyan] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if cmd == "quit" or cmd == "exit":
            console.print("[dim]Goodbye.[/dim]")
            break

        elif cmd == "show":
            if prd_content:
                console.print(Panel(prd_content, title="Current PRD"))
            else:
                console.print("[yellow]PRD is empty. Use 'append' or 'replace' to add content.[/yellow]")

        elif cmd == "append":
            console.print("[dim]Enter text (blank line to finish):[/dim]")
            lines: list[str] = []
            while True:
                try:
                    line = console.input("")
                except (EOFError, KeyboardInterrupt):
                    break
                if line == "":
                    break
                lines.append(line)
            if lines:
                new_text = "\n".join(lines)
                prd_content = (prd_content + "\n" + new_text).strip()
                console.print(f"[green]Appended {len(new_text)} characters.[/green]")

        elif cmd == "replace":
            console.print("[dim]Enter new PRD content (blank line to finish):[/dim]")
            lines = []
            while True:
                try:
                    line = console.input("")
                except (EOFError, KeyboardInterrupt):
                    break
                if line == "":
                    break
                lines.append(line)
            if lines:
                prd_content = "\n".join(lines)
                console.print(f"[green]PRD replaced ({len(prd_content)} characters).[/green]")

        elif cmd == "save":
            save_path = project_dir / "PRD.md"
            save_path.write_text(prd_content, encoding="utf-8")
            console.print(f"[green]PRD saved to {save_path}[/green]")

        elif cmd == "plan":
            if not prd_content:
                console.print("[red]PRD is empty. Add content first.[/red]")
                continue
            _load_env()
            _setup_logging()

            async def _plan() -> Any:
                from eunomia_v2.agents.planner import generate_task_graph
                return await generate_task_graph(prd_content)

            console.print("\n[bold]Generating task graph...[/bold]")
            try:
                task_graph = asyncio.run(_plan())
                table = Table(title="Generated Tasks")
                table.add_column("ID", style="cyan", width=4)
                table.add_column("Type", style="magenta", width=16)
                table.add_column("Title", style="white")
                for task in task_graph.tasks:
                    table.add_row(str(task.id), task.task_type.value, task.title)
                console.print(table)
            except Exception as e:
                console.print(f"[red]Planner failed: {e}[/red]")

        elif cmd == "run":
            if not prd_content:
                console.print("[red]PRD is empty. Add content first.[/red]")
                continue
            _load_env()
            _setup_logging()
            session_id = str(uuid.uuid4())[:8]
            console.print(f"[dim]Starting pipeline (session {session_id})...[/dim]")
            try:
                _run_pipeline_streaming(
                    project_dir, prd_content, DEFAULT_MODEL, 3, session_id,
                    verbose=False, hitl_level="approval",
                )
            except SystemExit:
                pass  # typer.Exit from pipeline errors

        elif cmd.startswith("generate"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                from eunomia_v2.documents.loader import list_templates
                available = list_templates()
                console.print(
                    f"[yellow]Usage: generate <doc_type>[/yellow]\n"
                    f"[dim]Available: {', '.join(available)}[/dim]"
                )
                continue

            gen_doc_type = parts[1].strip()
            _load_env()
            _setup_logging()

            from eunomia_v2.documents.loader import list_templates as lt
            from eunomia_v2.documents.loader import load_template, validate_context

            if gen_doc_type not in lt():
                console.print(f"[red]Unknown doc type: {gen_doc_type}[/red]")
                continue

            template = load_template(gen_doc_type)
            gen_ctx: dict[str, Any] = {}
            gen_ctx = _auto_discover_context(project_dir, gen_ctx)

            # Prompt for missing required context
            missing = validate_context(template, gen_ctx)
            for field in missing:
                req = next((r for r in template.required_context if r.name == field), None)
                desc = req.description if req else field
                try:
                    value = console.input(f"  [cyan]{field}[/cyan] ({desc}): ")
                    gen_ctx[field] = value
                except (EOFError, KeyboardInterrupt):
                    break

            async def _gen() -> str:
                from eunomia_v2.documents.generator import generate_document
                doc = await generate_document(
                    doc_type=gen_doc_type,
                    context=gen_ctx,
                    model=DEFAULT_MODEL,
                    review=True,
                )
                return doc.to_markdown()

            console.print(f"\n[bold]Generating {gen_doc_type}...[/bold]")
            try:
                gen_content = asyncio.run(_gen())
                docs_dir = project_dir / "docs"
                docs_dir.mkdir(parents=True, exist_ok=True)
                gen_path = docs_dir / f"{gen_doc_type}.md"
                gen_path.write_text(gen_content, encoding="utf-8")
                console.print(f"[green]Document saved to {gen_path}[/green]")
            except Exception as e:
                console.print(f"[red]Generation failed: {e}[/red]")

        else:
            console.print(f"[yellow]Unknown command: {cmd}. Type 'quit' to exit.[/yellow]")


# ---------------------------------------------------------------
# generate command — document generation
# ---------------------------------------------------------------

@app.command()
def generate(
    doc_type: str = typer.Argument(
        ..., help="Document type: prd, tdd, adr, readme, test_plan, runbook",
    ),
    project_path: str = typer.Option(".", "--project-path", help="Project working directory"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="LLM model to use"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    no_review: bool = typer.Option(False, "--no-review", help="Skip review pass"),
    context: list[str] = typer.Option(
        [], "--context", "-c", help="Context key=value pair",
    ),
) -> None:
    """Generate a document (PRD, TDD, ADR, README, test plan, runbook)."""
    _load_env()
    _setup_logging()

    from eunomia_v2.documents.loader import list_templates, load_template, validate_context

    # Validate doc_type
    available = list_templates()
    if doc_type not in available:
        console.print(f"[red]Unknown doc type: {doc_type}. Available: {', '.join(available)}[/red]")
        raise typer.Exit(1)

    # Parse context key=value pairs
    ctx: dict[str, Any] = {}
    for item in context:
        if "=" not in item:
            console.print(f"[red]Invalid context format: '{item}'. Use key=value.[/red]")
            raise typer.Exit(1)
        key, _, value = item.partition("=")
        ctx[key.strip()] = value.strip()

    # Auto-discover context from project
    project_dir = Path(project_path).resolve()
    ctx = _auto_discover_context(project_dir, ctx)

    # Check required context
    template = load_template(doc_type)
    missing = validate_context(template, ctx)
    if missing:
        console.print(f"[yellow]Missing required context: {', '.join(missing)}[/yellow]")
        console.print("[dim]Provide them with -c key=value flags.[/dim]")

        # Interactive prompt for missing values
        for field in missing:
            req = next((r for r in template.required_context if r.name == field), None)
            desc = req.description if req else field
            try:
                value = console.input(f"  [cyan]{field}[/cyan] ({desc}): ")
                ctx[field] = value
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Cancelled.[/dim]")
                raise typer.Exit(130)

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        docs_dir = project_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        output_path = docs_dir / f"{doc_type}.md"

    console.print(Panel(
        f"[bold]Type:[/bold] {doc_type}\n"
        f"[bold]Template:[/bold] {template.name}\n"
        f"[bold]Sections:[/bold] {template.get_section_count()}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Review:[/bold] {'yes' if not no_review else 'no'}\n"
        f"[bold]Output:[/bold] {output_path}",
        title="[bold blue]Eunomia V2 — Generate Document[/bold blue]",
    ))

    async def _generate() -> str:
        from eunomia_v2.documents.generator import generate_document
        doc = await generate_document(
            doc_type=doc_type,
            context=ctx,
            model=model,
            review=not no_review,
        )
        return doc.to_markdown()

    console.print("\n[bold]Generating document...[/bold]\n")
    try:
        content = asyncio.run(_generate())
    except Exception as e:
        console.print(f"[red]Generation failed: {e}[/red]")
        raise typer.Exit(1)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    console.print(f"\n[green]Document saved to {output_path}[/green]")


def _auto_discover_context(project_dir: Path, ctx: dict[str, Any]) -> dict[str, Any]:
    """Auto-discover context values from project files."""
    result = dict(ctx)

    # Try to get product_name/project_name from pyproject.toml
    pyproject = project_dir / "pyproject.toml"
    if pyproject.exists() and "product_name" not in result and "project_name" not in result:
        try:
            import tomllib
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            name = data.get("project", {}).get("name", "")
            if name:
                result.setdefault("product_name", name)
                result.setdefault("project_name", name)
                result.setdefault("system_name", name)
        except Exception:
            pass

    # Try to get product_vision from PRD.md
    for prd_name in ["PRD.md", "prd.md"]:
        prd_path = project_dir / prd_name
        if prd_path.exists() and "product_vision" not in result:
            try:
                prd_text = prd_path.read_text(encoding="utf-8")[:2000]
                result.setdefault("product_vision", prd_text)
                result.setdefault("project_description", prd_text[:500])
            except Exception:
                pass
            break

    return result


# ---------------------------------------------------------------
# sessions command — list persisted sessions
# ---------------------------------------------------------------

@app.command()
def sessions(
    limit: int = typer.Option(20, "--limit", help="Max sessions to display"),
) -> None:
    """List recent pipeline sessions from SQLite."""
    from eunomia_v2.persistence.session_store import SQLiteSessionStore

    store = SQLiteSessionStore()
    rows = store.list_sessions(limit=limit)

    if not rows:
        console.print("[yellow]No sessions found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Recent Sessions (limit={limit})")
    table.add_column("Session ID", style="cyan", width=10)
    table.add_column("Status", width=10)
    table.add_column("Model", style="dim", width=20)
    table.add_column("Tasks", width=8)
    table.add_column("Done", width=6)
    table.add_column("Fail", width=6)
    table.add_column("Project", style="dim")
    table.add_column("Created", style="dim", width=20)

    for row in rows:
        status_str = row.get("status", "")
        if status_str == "completed":
            status_display = f"[green]{status_str}[/green]"
        elif status_str == "failed":
            status_display = f"[red]{status_str}[/red]"
        elif status_str == "running":
            status_display = f"[yellow]{status_str}[/yellow]"
        else:
            status_display = status_str

        completed = row.get("completed_tasks", [])
        failed = row.get("failed_tasks", [])

        table.add_row(
            row.get("session_id", "")[:10],
            status_display,
            row.get("model", "")[:20],
            str(row.get("total_tasks", 0)),
            str(len(completed)),
            str(len(failed)),
            str(row.get("project_path", ""))[-30:],
            row.get("created_at", "")[:19],
        )

    console.print(table)


# ---------------------------------------------------------------
# resume command — resume an interrupted pipeline
# ---------------------------------------------------------------

@app.command()
def resume(
    project_path: str = typer.Option(".", "--project-path", help="Project working directory"),
    session_id: str = typer.Option(None, "--session-id", help="Session ID to resume"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="LLM model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Resume a previously interrupted pipeline from SQLite checkpoint."""
    _load_env()
    _setup_logging(verbose)

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    project_dir = Path(project_path).resolve()

    from eunomia_v2.persistence.session_store import SQLiteSessionStore

    store = SQLiteSessionStore()

    # If no session_id provided, find most recent session for this project
    if not session_id:
        all_sessions = store.list_sessions(limit=50)
        project_sessions = [
            s for s in all_sessions
            if s.get("project_path", "") == str(project_dir)
        ]
        if not project_sessions:
            console.print(f"[red]No sessions found for project {project_dir}[/red]")
            raise typer.Exit(1)
        session_data = project_sessions[0]  # Most recent
        session_id = session_data["session_id"]
        console.print(f"[dim]Resuming most recent session: {session_id}[/dim]")
    else:
        session_data = store.load_session(session_id)
        if session_data is None:
            console.print(f"[red]Session {session_id} not found in SQLite.[/red]")
            raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Session:[/bold] {session_id}\n"
        f"[bold]Project:[/bold] {session_data.get('project_path', '')}\n"
        f"[bold]Status:[/bold] {session_data.get('status', '')}\n"
        f"[bold]Model:[/bold] {model}",
        title="[bold blue]Eunomia V2 — Resume Pipeline[/bold blue]",
    ))

    async def _resume() -> dict[str, Any]:
        from langgraph.types import Command

        from eunomia_v2.cli.renderer import PipelineRenderer
        from eunomia_v2.graph.orchestrator import compile_graph
        from eunomia_v2.graph.streaming import StreamEvent
        from eunomia_v2.persistence.checkpointer import get_checkpointer

        checkpointer = get_checkpointer("sqlite")
        graph = compile_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": f"cli-{session_id}"}}

        renderer = PipelineRenderer(console=console, verbose=verbose)

        try:
            # Check current state from checkpoint
            snapshot = await graph.aget_state(config)
            if snapshot is None or not snapshot.values:
                console.print("[red]No checkpoint found for this session.[/red]")
                return {}

            # If there's a pending interrupt, prompt user and resume
            if snapshot.tasks:
                console.print("[yellow]Pipeline has a pending interrupt.[/yellow]")
                user_input = console.input("[bold yellow]> Your response: [/bold yellow]")

                from eunomia_v2.graph.streaming import resume_pipeline
                async for event in resume_pipeline(graph, user_input, config):
                    renderer.render(event)

                # Handle subsequent interrupts
                while renderer.pending_interrupt is not None:
                    user_input = console.input("[bold yellow]> Your response: [/bold yellow]")
                    renderer.pending_interrupt = None
                    async for event in resume_pipeline(graph, user_input, config):
                        renderer.render(event)

            final_snapshot = await graph.aget_state(config)
            return final_snapshot.values if final_snapshot else {}
        finally:
            await _close_checkpointer(checkpointer)

    try:
        final_state = asyncio.run(_resume())
    except KeyboardInterrupt:
        console.print("\n[yellow]Resume interrupted by user.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Resume failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)

    if not final_state:
        return

    # Update session store with final results
    completed = list(final_state.get("completed_tasks", []))
    failed = list(final_state.get("failed_tasks", []))
    total = len(final_state.get("tasks", []))

    store.update_session(
        session_id,
        status="completed" if not failed else "failed",
        total_tasks=total,
        completed_tasks=completed,
        failed_tasks=failed,
    )

    console.print(Panel(
        f"[bold]Total:[/bold] {total}  |  "
        f"[green]Completed:[/green] {len(completed)}  |  "
        f"[red]Failed:[/red] {len(failed)}",
        title=f"[bold blue]Resume Complete — {session_id}[/bold blue]",
    ))


# ---------------------------------------------------------------
# serve command
# ---------------------------------------------------------------

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", help="Server port"),
) -> None:
    """Start the FastAPI server."""
    console.print(f"[bold blue]Eunomia V2[/bold blue] — Starting server on {host}:{port}")
    import uvicorn

    uvicorn.run("eunomia_v2.api.server:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
