from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


ROOT = Path(__file__).resolve().parent.parent
SOURCE_REPO = ROOT.parent
PROJECT_ROOT = SOURCE_REPO.parent
CONFIG_DIR = ROOT / "config"
SCRIPTS_DIR = ROOT / "scripts"
BENCH_PROFILES_PATH = CONFIG_DIR / "bench_profiles.json"
BENCH_LOCAL_PATH = CONFIG_DIR / "bench_local.json"
JAVA_HELPER_PATH = SCRIPTS_DIR / "java_env_utils.sh"
SETUP_ACTIONS = {
    "bootstrap_venv",
    "prepare_dataset",
    "build_runtime_archive",
    "extract_runtime",
    "extract_runtime_force",
    "prepare_runtime",
    "import_dataset",
    "import_dataset_reset",
    "start_server",
    "wait_for_index_online",
}
SHELL_COMMANDS = {"doctor", "status", "setup", "start", "stop", "run", "results"}
COMMAND_SUMMARIES = {
    "status": "Show the current benchmark workspace state and what is ready.",
    "doctor": "Run the same checks as status and add concrete fix suggestions.",
    "setup": "Prepare missing dataset/runtime pieces without running a benchmark.",
    "start": "Start the current Neo4j benchmark runtime and wait for the index.",
    "stop": "Stop the current Neo4j benchmark runtime if it is running.",
    "run": "Run a named benchmark mode against a selected workload tier.",
    "results": "Show the latest benchmark results or browse recent result summaries.",
}
COMMAND_DETAILS = {
    "status": {
        "usage": "status",
        "details": [
            "Shows dataset, runtime, Bolt, import-marker, and vector-index readiness.",
            "Useful before and after setup, and before a long benchmark run.",
        ],
        "examples": ["status"],
    },
    "doctor": {
        "usage": "doctor",
        "details": [
            "Runs the same environment inspection as status.",
            "Adds actionable messages for missing Java, dataset artifacts, runtime prep, import mismatch, and index readiness.",
        ],
        "examples": ["doctor"],
    },
    "setup": {
        "usage": "setup [--fresh] [--reimport] [--rebuild-runtime]",
        "details": [
            "Builds only what is missing by default.",
            "--reimport replaces only the imported ogbn-arxiv database.",
            "--rebuild-runtime rebuilds and re-extracts the packaged Neo4j runtime.",
            "--fresh does the whole pipeline again: dataset prep, runtime rebuild, and reimport.",
        ],
        "examples": ["setup", "setup --reimport", "setup --fresh"],
    },
    "start": {
        "usage": "start",
        "details": [
            "Starts the resolved Neo4j runtime.",
            "If setup is incomplete, it prepares missing prerequisites first.",
            "Waits for Bolt and the vector index before returning.",
        ],
        "examples": ["start"],
    },
    "stop": {
        "usage": "stop",
        "details": [
            "Stops the resolved Neo4j runtime if it is currently running.",
            "Safe to call repeatedly.",
        ],
        "examples": ["stop"],
    },
    "run": {
        "usage": "run --tier smoke|dev|full --mode rerank|parity|native-only|baseline-only [--order native-first|baseline-first]",
        "details": [
            "Auto-runs setup checks before executing the benchmark.",
            "rerank and parity are comparison runs; native-only and baseline-only run one method.",
            "Comparison mode will restart Neo4j between phases to reduce cache bias.",
        ],
        "examples": [
            "run --tier smoke --mode rerank",
            "run --tier smoke --mode parity",
            "run --tier dev --mode native-only",
        ],
    },
    "results": {
        "usage": "results [--recent N] [--file PATH] [--kind comparison|method|all]",
        "details": [
            "Without arguments, shows the latest comparison summary if one exists, otherwise the latest summary artifact.",
            "--recent lists the newest result artifacts without expanding all metrics.",
            "--file opens one specific summary JSON and renders it in the shell.",
            "--kind filters the artifact search to comparison summaries, method summaries, or both.",
        ],
        "examples": [
            "results",
            "results --recent 5",
            "results --kind comparison",
            "results --file results/summaries/comparison-20260414-185914.json",
        ],
    },
}
MODE_DESCRIPTIONS = {
    "rerank": "Comparison run with the normal rag.retrieve reranking defaults.",
    "parity": "Comparison run with reranking effectively disabled to check native vs baseline agreement.",
    "native-only": "Run only CALL rag.retrieve(...) using the rerank defaults.",
    "baseline-only": "Run only the two-call baseline workflow.",
}
TIER_DESCRIPTIONS = {
    "smoke": "100 sampled papers x 12 scenarios = 1200 cases. Best first validation run.",
    "dev": "500 sampled papers x 12 scenarios = 6000 cases. Good day-to-day benchmark tier.",
    "full": "2000 sampled papers x 12 scenarios = 24000 cases. Longest local run.",
}
SETUP_ACTION_DETAILS = {
    "bootstrap_venv": "Create the Python environment used by the benchmark workspace.",
    "prepare_dataset": "Download/process ogbn-arxiv and generate manifests.",
    "build_runtime_archive": "Create the packaged Neo4j native-rag tarball using a downloaded/cached stock archive plus the plugin jar.",
    "extract_runtime": "Extract the packaged runtime into the local runtime workspace.",
    "extract_runtime_force": "Re-extract the packaged runtime, replacing the current extracted runtime.",
    "prepare_runtime": "Copy the plugin jar and append benchmark config overrides.",
    "import_dataset": "Offline-import ogbn-arxiv into the current runtime and verify the index.",
    "import_dataset_reset": "Replace the current imported ogbn-arxiv database and verify the index.",
    "start_server": "Start Neo4j and wait for Bolt readiness.",
    "wait_for_index_online": "Wait for the vector index to become ONLINE before continuing.",
}
SHELL_HISTORY_PATH = ROOT / ".bench_history"
PROMPT_STYLE = PromptStyle.from_dict(
    {
        "prompt.prefix": "bold ansicyan",
        "prompt.dataset": "bold ansimagenta",
        "prompt.arrow": "bold ansigreen",
        "bottom-toolbar": "bg:#0f172a #cbd5e1",
    }
)


@dataclass(frozen=True)
class ModeProfile:
    name: str
    comparison: bool
    defaults_path: Path
    default_order: str | None
    method: str | None


@dataclass(frozen=True)
class BenchSettings:
    benchmark_root: Path
    source_repo: Path
    project_root: Path
    profiles_path: Path
    local_overrides_path: Path
    local_overrides: dict[str, Any]
    dataset: str
    runtime_version: str
    runtime_root: Path
    runtime_home: Path
    runtime_archive: Path
    uri: str
    database: str
    user: str
    password: str
    results_root: Path
    index_name: str
    processed_manifest_path: Path
    processed_dir: Path
    tier_manifests: dict[str, Path]
    mode_profiles: dict[str, ModeProfile]
    java_home_override: str | None


@dataclass(frozen=True)
class EnvironmentState:
    venv_ready: bool
    dataset_ready: bool
    runtime_archive_ready: bool
    runtime_extracted: bool
    runtime_prepared: bool
    import_state: str
    server_running: bool
    bolt_ready: bool
    index_state: str | None
    index_ready: bool
    local_overrides_present: bool


def format_kv_lines(items: list[tuple[str, str]], *, indent: str = "  ") -> list[str]:
    width = max((len(label) for label, _ in items), default=0)
    return [f"{indent}{label.ljust(width)} : {value}" for label, value in items]


def shell_prompt(settings: BenchSettings) -> str:
    return f"bench[{settings.dataset}]> "


def check_label(ready: bool, *, missing_label: str = "missing") -> str:
    return "ready" if ready else missing_label


def next_step_hint(state: EnvironmentState) -> str:
    if not state.venv_ready or not state.dataset_ready or not state.runtime_archive_ready or not state.runtime_extracted:
        return "Run `setup` to build the missing benchmark prerequisites."
    if not state.runtime_prepared:
        return "Run `setup` to copy the plugin and runtime overrides."
    if state.import_state == "missing":
        return "Run `setup` to import ogbn-arxiv into the runtime."
    if state.import_state == "mismatch":
        return "Run `setup --reimport` to refresh the imported database."
    if not state.server_running:
        return "Run `start` or let `run ...` start Neo4j automatically."
    if not state.bolt_ready:
        return "Neo4j is starting. Wait a moment, then run `status` again."
    if not state.index_ready:
        return "Wait for the vector index to become ONLINE, or run `start` again."
    return "Environment is ready. Run `run --tier smoke --mode rerank` to start benchmarking."


def console(*, stderr: bool = False) -> Console:
    return Console(file=sys.stderr if stderr else sys.stdout, highlight=False, soft_wrap=True)


def status_style(value: str) -> str:
    if value in {"ready", "ONLINE", "yes"}:
        return "bold green"
    if value in {"missing", "mismatch", "no", "unknown"}:
        return "bold yellow"
    return "bold cyan"


def make_info_panel(title: str, body: str, *, border_style: str = "cyan") -> Panel:
    return Panel(body, title=title, border_style=border_style, expand=False, padding=(0, 1))


def command_completions() -> list[str]:
    completions = [
        "status",
        "doctor",
        "setup",
        "setup --fresh",
        "setup --reimport",
        "setup --rebuild-runtime",
        "start",
        "stop",
        "results",
        "results --recent 5",
        "results --kind comparison",
        "results --kind method",
        "run --tier smoke --mode rerank",
        "run --tier smoke --mode parity",
        "run --tier dev --mode rerank",
        "run --tier dev --mode native-only",
        "run --tier dev --mode baseline-only",
        "help",
        "help run",
        "help setup",
        "help modes",
        "help tiers",
        "exit",
        "quit",
    ]
    return completions


def build_prompt_session() -> PromptSession[str]:
    return PromptSession(
        history=FileHistory(str(SHELL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(command_completions(), ignore_case=True, sentence=True),
        style=PROMPT_STYLE,
    )


def prompt_message(settings: BenchSettings) -> HTML:
    return HTML(
        "<prompt.prefix>bench</prompt.prefix>"
        f"[<prompt.dataset>{settings.dataset}</prompt.dataset>]"
        "<prompt.arrow>> </prompt.arrow>"
    )


def prompt_toolbar() -> HTML:
    return HTML(" help  |  doctor  |  setup  |  run --tier smoke --mode rerank  |  exit ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teammate-friendly benchmark CLI for the Neo4j RAG workspace.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runtime-root", help="Override the runtime workspace root directory.")
    common.add_argument("--runtime-home", help="Override the extracted Neo4j runtime directory directly.")
    common.add_argument("--uri", help="Override the Bolt URI for benchmark and health checks.")
    common.add_argument("--database", help="Override the Neo4j database name.")
    common.add_argument("--java-home", help="Override JAVA_HOME for Neo4j CLI commands.")
    common.add_argument("--results-root", help="Override the benchmark results root directory.")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("shell", parents=[common], help="Enter the interactive benchmark shell.")
    subparsers.add_parser("doctor", parents=[common], help="Check local benchmark readiness and print fix hints.")
    subparsers.add_parser("status", parents=[common], help="Show the resolved runtime and benchmark environment state.")

    setup_parser = subparsers.add_parser("setup", parents=[common], help="Prepare the benchmark runtime and dataset.")
    setup_parser.add_argument("--fresh", action="store_true", help="Rebuild dataset artifacts, runtime, and import.")
    setup_parser.add_argument("--reimport", action="store_true", help="Replace only the imported ogbn-arxiv database.")
    setup_parser.add_argument("--rebuild-runtime", action="store_true", help="Rebuild and re-extract the packaged runtime.")

    subparsers.add_parser("start", parents=[common], help="Start the resolved benchmark runtime.")
    subparsers.add_parser("stop", parents=[common], help="Stop the resolved benchmark runtime if it is running.")
    results_parser = subparsers.add_parser("results", parents=[common], help="Show the latest benchmark results.")
    results_parser.add_argument("--recent", type=int, default=0, help="List the newest N summary artifacts instead of expanding one summary.")
    results_parser.add_argument("--file", help="Render a specific summary JSON file.")
    results_parser.add_argument(
        "--kind",
        choices=("comparison", "method", "all"),
        default="all",
        help="Filter result artifacts by summary type.",
    )

    run_parser = subparsers.add_parser("run", parents=[common], help="Run a benchmark tier in one named mode.")
    run_parser.add_argument("--tier", required=True, choices=("smoke", "dev", "full"))
    run_parser.add_argument(
        "--mode",
        required=True,
        choices=("rerank", "parity", "native-only", "baseline-only"),
    )
    run_parser.add_argument("--order", choices=("native-first", "baseline-first"), help="Override the comparison order.")
    run_parser.add_argument("--fresh", action="store_true", help="Rebuild dataset artifacts, runtime, and import.")
    run_parser.add_argument("--reimport", action="store_true", help="Replace only the imported ogbn-arxiv database.")
    run_parser.add_argument("--rebuild-runtime", action="store_true", help="Rebuild and re-extract the packaged runtime.")
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "shell"
    return args


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load_json(path: Path) -> dict[str, Any]:
    return load_json(path) if path.exists() else {}


def resolve_path(value: str, base: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def path_for_display(path: Path, *, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def normalize_string(value: Any) -> str:
    return "" if value is None else str(value)


def build_mode_profiles(raw_profiles: dict[str, Any], *, benchmark_root: Path) -> dict[str, ModeProfile]:
    modes: dict[str, ModeProfile] = {}
    for name, raw in raw_profiles.items():
        modes[name] = ModeProfile(
            name=name,
            comparison=bool(raw.get("comparison", False)),
            defaults_path=resolve_path(str(raw["defaults"]), benchmark_root),
            default_order=raw.get("order"),
            method=raw.get("method"),
        )
    return modes


def resolve_settings(
    args: argparse.Namespace,
    *,
    benchmark_root: Path = ROOT,
    profiles_path: Path = BENCH_PROFILES_PATH,
    local_overrides_path: Path = BENCH_LOCAL_PATH,
) -> BenchSettings:
    source_repo = benchmark_root.parent
    project_root = source_repo.parent
    profiles = load_json(profiles_path)
    defaults = dict(profiles.get("defaults", {}))
    local_overrides = maybe_load_json(local_overrides_path)

    def pick(key: str) -> Any:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            return cli_value
        if key in local_overrides:
            return local_overrides[key]
        return defaults.get(key)

    runtime_version = normalize_string(pick("runtime_version") or defaults["runtime_version"])
    runtime_root = resolve_path(normalize_string(pick("runtime_root") or defaults["runtime_root"]), benchmark_root)
    runtime_home_override = getattr(args, "runtime_home", None) or local_overrides.get("runtime_home")
    if runtime_home_override:
        runtime_home = resolve_path(normalize_string(runtime_home_override), benchmark_root)
    else:
        runtime_home = (runtime_root / f"neo4j-community-{runtime_version}-native-rag").resolve()

    runtime_archive = (
        source_repo / "packaging" / "standalone" / "target" / f"neo4j-community-{runtime_version}-native-rag.tar.gz"
    ).resolve()
    tier_manifests = {
        tier: resolve_path(str(entry["query_manifest"]), benchmark_root)
        for tier, entry in profiles.get("tiers", {}).items()
    }

    return BenchSettings(
        benchmark_root=benchmark_root,
        source_repo=source_repo,
        project_root=project_root,
        profiles_path=profiles_path,
        local_overrides_path=local_overrides_path,
        local_overrides=local_overrides,
        dataset=normalize_string(defaults["dataset"]),
        runtime_version=runtime_version,
        runtime_root=runtime_root,
        runtime_home=runtime_home,
        runtime_archive=runtime_archive,
        uri=normalize_string(pick("uri") or defaults["uri"]),
        database=normalize_string(pick("database") or defaults["database"]),
        user=normalize_string(local_overrides.get("user", "")),
        password=normalize_string(local_overrides.get("password", "")),
        results_root=resolve_path(normalize_string(pick("results_root") or defaults["results_root"]), benchmark_root),
        index_name=normalize_string(defaults["index_name"]),
        processed_manifest_path=resolve_path(str(defaults["processed_manifest"]), benchmark_root),
        processed_dir=resolve_path(str(defaults["processed_dir"]), benchmark_root),
        tier_manifests=tier_manifests,
        mode_profiles=build_mode_profiles(profiles.get("modes", {}), benchmark_root=benchmark_root),
        java_home_override=normalize_string(getattr(args, "java_home", None) or local_overrides.get("java_home", "")) or None,
    )


def venv_python_path(settings: BenchSettings) -> Path:
    return settings.benchmark_root / ".venv" / "bin" / "python"


def expected_runtime_config_lines(settings: BenchSettings) -> list[str]:
    override_path = settings.benchmark_root / "config" / "neo4j-rag.conf"
    if not override_path.exists():
        return []
    lines = []
    for raw_line in override_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def load_dataset_manifest(settings: BenchSettings) -> dict[str, Any] | None:
    if not settings.processed_manifest_path.exists():
        return None
    return load_json(settings.processed_manifest_path)


def dataset_artifacts_ready(settings: BenchSettings) -> bool:
    manifest = load_dataset_manifest(settings)
    if manifest is None:
        return False

    required_paths = [settings.processed_manifest_path]
    files = manifest.get("files", {})
    for key in ("papers_csv", "cites_csv"):
        relative_path = files.get(key, {}).get("path")
        if not relative_path:
            return False
        required_paths.append(resolve_path(str(relative_path), settings.benchmark_root))
    required_paths.extend(settings.tier_manifests.values())
    return all(path.exists() for path in required_paths)


def runtime_prepared(settings: BenchSettings) -> bool:
    if not settings.runtime_home.exists():
        return False

    plugin_dir = settings.runtime_home / "plugins"
    conf_path = settings.runtime_home / "conf" / "neo4j.conf"
    if not plugin_dir.exists() or not conf_path.exists():
        return False
    if not list(plugin_dir.glob("neo4j-rag-plugin-*.jar")):
        return False

    conf_text = conf_path.read_text(encoding="utf-8")
    return all(line in conf_text for line in expected_runtime_config_lines(settings))


def import_marker_path(settings: BenchSettings) -> Path:
    return settings.runtime_home / ".rag-benchmark" / settings.database / "ogbn-arxiv-import.json"


def import_state(settings: BenchSettings) -> str:
    marker_path = import_marker_path(settings)
    if not marker_path.exists():
        return "missing"

    manifest = load_dataset_manifest(settings)
    if manifest is None:
        return "mismatch"

    marker = load_json(marker_path)
    if marker.get("dataset_name") != settings.dataset:
        return "mismatch"
    if marker.get("database") != settings.database:
        return "mismatch"
    if marker.get("index_name") != settings.index_name:
        return "mismatch"
    if marker.get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
        return "mismatch"
    return "ready"


def resolve_java_home(settings: BenchSettings) -> str | None:
    if settings.java_home_override:
        candidate = Path(settings.java_home_override)
        if (candidate / "bin" / "java").exists():
            return str(candidate)
        return None

    command = (
        f"source {shlex.quote(str(JAVA_HELPER_PATH))} >/dev/null 2>&1 && "
        "resolve_java_home"
    )
    result = subprocess.run(
        ["/bin/bash", "-lc", command],
        cwd=settings.benchmark_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    resolved = result.stdout.strip()
    return resolved or None


def command_env(settings: BenchSettings, *, require_java: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    java_home = resolve_java_home(settings)
    if java_home:
        env["JAVA_HOME"] = java_home
        env["PATH"] = f"{java_home}/bin:{env.get('PATH', '')}"
    elif require_java:
        raise RuntimeError("Unable to resolve JAVA_HOME. Install Java 17 or 21 or set config/bench_local.json.")
    return env


def run_command(
    command: list[str],
    *,
    settings: BenchSettings,
    cwd: Path | None = None,
    require_java: bool = False,
    extra_env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = command_env(settings, require_java=require_java)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        command,
        cwd=cwd or settings.benchmark_root,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


def stream_command(
    command: list[str],
    *,
    settings: BenchSettings,
    cwd: Path | None = None,
    require_java: bool = False,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    env = command_env(settings, require_java=require_java)
    if extra_env:
        env.update(extra_env)

    process = subprocess.Popen(
        command,
        cwd=cwd or settings.benchmark_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None

    lines: list[str] = []
    for raw_line in process.stdout:
        console().print(Text(raw_line.rstrip("\n")))
        lines.append(raw_line.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return lines


def server_running(settings: BenchSettings) -> bool:
    if not settings.runtime_home.exists():
        return False
    try:
        result = run_command(
            [str(settings.runtime_home / "bin" / "neo4j"), "status"],
            settings=settings,
            require_java=True,
            check=False,
        )
    except RuntimeError:
        return False
    return result.returncode == 0


def cypher_scalar(settings: BenchSettings, query: str) -> str | None:
    if not settings.runtime_home.exists():
        return None

    command = [
        str(settings.runtime_home / "bin" / "cypher-shell"),
        "-a",
        settings.uri,
        "-d",
        settings.database,
        "--non-interactive",
        "--format",
        "plain",
    ]
    if settings.user:
        command.extend(["-u", settings.user, "-p", settings.password])
    command.append(query)

    try:
        result = run_command(command, settings=settings, require_java=True, check=False)
    except RuntimeError:
        return None
    if result.returncode != 0:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line == "value" and index + 1 < len(lines):
            value = lines[index + 1].strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value

    for line in reversed(lines):
        if re.fullmatch(r"\d+\s+rows?", line):
            continue
        value = line
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value
    return None


def wait_for_bolt(settings: BenchSettings, *, timeout_seconds: int = 60) -> None:
    for _ in range(timeout_seconds):
        if cypher_scalar(settings, "RETURN 1 AS value;") == "1":
            return
        time_sleep()
    raise RuntimeError(f"Neo4j Bolt endpoint at {settings.uri} did not become ready within {timeout_seconds} seconds.")


def wait_for_index_online(settings: BenchSettings, *, timeout_seconds: int = 60) -> None:
    for _ in range(timeout_seconds):
        if index_state(settings) == "ONLINE":
            return
        time_sleep()
    raise RuntimeError(f"Vector index {settings.index_name} did not become ONLINE within {timeout_seconds} seconds.")


def time_sleep(seconds: float = 1.0) -> None:
    import time

    time.sleep(seconds)


def bolt_ready(settings: BenchSettings) -> bool:
    return cypher_scalar(settings, "RETURN 1 AS value;") == "1"


def index_state(settings: BenchSettings) -> str | None:
    query = (
        "SHOW VECTOR INDEXES YIELD name, state "
        f"WHERE name = '{settings.index_name}' RETURN state AS value;"
    )
    return cypher_scalar(settings, query)


def inspect_environment(settings: BenchSettings) -> EnvironmentState:
    is_running = server_running(settings)
    is_bolt_ready = bolt_ready(settings) if is_running else False
    current_index_state = index_state(settings) if is_bolt_ready else None
    return EnvironmentState(
        venv_ready=venv_python_path(settings).exists(),
        dataset_ready=dataset_artifacts_ready(settings),
        runtime_archive_ready=settings.runtime_archive.exists(),
        runtime_extracted=settings.runtime_home.exists(),
        runtime_prepared=runtime_prepared(settings),
        import_state=import_state(settings),
        server_running=is_running,
        bolt_ready=is_bolt_ready,
        index_state=current_index_state,
        index_ready=current_index_state == "ONLINE",
        local_overrides_present=settings.local_overrides_path.exists(),
    )


def compute_setup_actions(
    state: EnvironmentState,
    *,
    fresh: bool,
    reimport: bool,
    rebuild_runtime: bool,
    ensure_server: bool,
) -> list[str]:
    actions: list[str] = []
    if fresh:
        reimport = True
        rebuild_runtime = True

    venv_ready = state.venv_ready
    dataset_ready = state.dataset_ready
    runtime_archive_ready = state.runtime_archive_ready
    runtime_extracted = state.runtime_extracted
    runtime_prepared_flag = state.runtime_prepared
    import_status = state.import_state
    running = state.server_running
    index_ready_flag = state.index_ready

    if not venv_ready:
        actions.append("bootstrap_venv")
        venv_ready = True

    if fresh or not dataset_ready:
        actions.append("prepare_dataset")
        dataset_ready = True
        import_status = "missing"

    if fresh or rebuild_runtime or not runtime_archive_ready:
        actions.append("build_runtime_archive")
        runtime_archive_ready = True

    if fresh or rebuild_runtime or not runtime_extracted:
        actions.append("extract_runtime_force" if runtime_extracted else "extract_runtime")
        runtime_extracted = True
        runtime_prepared_flag = False
        import_status = "missing"
        running = False
        index_ready_flag = False

    if fresh or rebuild_runtime or not runtime_prepared_flag:
        actions.append("prepare_runtime")
        runtime_prepared_flag = True
        import_status = "missing"
        running = False
        index_ready_flag = False

    if fresh or reimport:
        actions.append("import_dataset_reset")
        import_status = "ready"
        running = True
        index_ready_flag = True
    elif import_status == "missing":
        actions.append("import_dataset")
        import_status = "ready"
        running = True
        index_ready_flag = True
    elif import_status == "mismatch":
        raise RuntimeError(
            "The current runtime import marker does not match the processed ogbn-arxiv manifest. "
            "Re-run with --reimport or --fresh."
        )

    if ensure_server:
        if not running:
            actions.append("start_server")
            running = True
        if not index_ready_flag:
            actions.append("wait_for_index_online")

    unexpected = [action for action in actions if action not in SETUP_ACTIONS]
    if unexpected:
        raise ValueError(f"Unexpected setup actions computed: {unexpected}")
    return actions


def print_action(name: str) -> None:
    console().print(f"[bold cyan]bench[/bold cyan] [dim]>[/dim] {name}")


def render_command_help(command: str) -> str:
    details = COMMAND_DETAILS[command]
    lines = [
        f"Command: {command}",
        f"Summary: {COMMAND_SUMMARIES[command]}",
        "",
        f"Usage: {details['usage']}",
        "",
        "What it does",
    ]
    for entry in details["details"]:
        lines.append(f"  - {entry}")

    lines.append("")
    lines.append("Examples")
    for example in details["examples"]:
        lines.append(f"  {example}")

    if command == "run":
        lines.append("")
        lines.append("Modes")
        for name, description in MODE_DESCRIPTIONS.items():
            lines.append(f"  {name.ljust(13)} {description}")
        lines.append("")
        lines.append("Tiers")
        for name, description in TIER_DESCRIPTIONS.items():
            lines.append(f"  {name.ljust(13)} {description}")

    return "\n".join(lines)


def render_modes_help() -> str:
    lines = ["Benchmark modes"]
    for name, description in MODE_DESCRIPTIONS.items():
        lines.append(f"  {name.ljust(13)} {description}")
    return "\n".join(lines)


def render_tiers_help() -> str:
    lines = ["Benchmark tiers"]
    for name, description in TIER_DESCRIPTIONS.items():
        lines.append(f"  {name.ljust(13)} {description}")
    return "\n".join(lines)


def render_shell_help(topic: str | None = None) -> str:
    normalized = (topic or "").strip().lower()
    if normalized in COMMAND_DETAILS:
        return render_command_help(normalized)
    if normalized == "modes":
        return render_modes_help()
    if normalized == "tiers":
        return render_tiers_help()
    if normalized:
        return f"Unknown help topic: {normalized}\nType `help` to see the command list."

    command_rows = [(name, COMMAND_SUMMARIES[name]) for name in ("status", "doctor", "setup", "start", "stop", "run", "results")]
    width = max(len(name) for name, _ in command_rows)
    lines = [
        "Neo4j RAG Benchmark Shell",
        "=========================",
        "",
        "Core commands",
    ]
    lines.extend([f"  {name.ljust(width)}  {summary}" for name, summary in command_rows])
    lines.extend(
        [
            "",
            "Extra help topics",
            "  help modes     Explain benchmark modes.",
            "  help tiers     Explain workload tiers.",
            "  help run       Show the full run command help.",
            "",
            "Shell controls",
            "  help           Show this overview.",
            "  exit           Leave the shell.",
            "  quit           Leave the shell.",
            "",
            "Quick start",
            "  doctor",
            "  setup",
            "  run --tier smoke --mode rerank",
            "  results",
        ]
    )
    return "\n".join(lines)


def render_shell_banner(settings: BenchSettings) -> str:
    return "\n".join(
        [
            "Neo4j RAG Benchmark Shell",
            "=========================",
            f"  dataset      : {settings.dataset}",
            f"  runtime_home : {settings.runtime_home}",
            f"  uri          : {settings.uri}",
            "",
            "Type `help` for commands, `help run` for the benchmark modes, and `exit` to leave.",
        ]
    )


def shell_override_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    for option_name, flag in (
        ("runtime_root", "--runtime-root"),
        ("runtime_home", "--runtime-home"),
        ("uri", "--uri"),
        ("database", "--database"),
        ("java_home", "--java-home"),
        ("results_root", "--results-root"),
    ):
        value = getattr(args, option_name, None)
        if value:
            argv.extend([flag, str(value)])
    return argv


def shell_command_argv(shell_args: argparse.Namespace, tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    if tokens[0] not in SHELL_COMMANDS:
        return tokens
    return [tokens[0], *shell_override_argv(shell_args), *tokens[1:]]


def import_script_env(settings: BenchSettings) -> dict[str, str]:
    env = {"NEO4J_URI": settings.uri}
    if settings.user:
        env["NEO4J_USER"] = settings.user
        env["NEO4J_PASSWORD"] = settings.password
    return env


def execute_setup_action(action: str, settings: BenchSettings) -> None:
    if action == "bootstrap_venv":
        print_action("Creating Python environment...")
        stream_command([str(settings.benchmark_root / "scripts" / "bootstrap_python.sh")], settings=settings)
        return

    if action == "prepare_dataset":
        print_action("Preparing ogbn-arxiv dataset artifacts...")
        stream_command(
            [str(venv_python_path(settings)), "-m", "benchmark.prepare_ogbn_arxiv"],
            settings=settings,
        )
        return

    if action == "build_runtime_archive":
        print_action("Building packaged Neo4j runtime archive...")
        stream_command(
            [str(settings.benchmark_root / "scripts" / "build_native_rag_tar.sh"), settings.runtime_version],
            settings=settings,
            require_java=True,
        )
        return

    if action in {"extract_runtime", "extract_runtime_force"}:
        print_action("Extracting packaged Neo4j runtime...")
        command = [
            str(settings.benchmark_root / "scripts" / "extract_native_rag_runtime.sh"),
            settings.runtime_version,
        ]
        if action == "extract_runtime_force":
            command.append("--force")
        stream_command(command, settings=settings)
        return

    if action == "prepare_runtime":
        print_action("Preparing extracted runtime with plugin and config...")
        stream_command(
            [str(settings.benchmark_root / "scripts" / "prepare_runtime.sh"), str(settings.runtime_home)],
            settings=settings,
            require_java=True,
        )
        return

    if action in {"import_dataset", "import_dataset_reset"}:
        print_action("Importing ogbn-arxiv into the benchmark runtime...")
        command = [str(settings.benchmark_root / "scripts" / "import_ogbn_arxiv.sh"), str(settings.runtime_home)]
        if action == "import_dataset_reset":
            command.append("--reset")
        stream_command(
            command,
            settings=settings,
            require_java=True,
            extra_env=import_script_env(settings),
        )
        return

    if action == "start_server":
        print_action("Starting Neo4j runtime...")
        stream_command(
            [str(settings.runtime_home / "bin" / "neo4j"), "start"],
            settings=settings,
            require_java=True,
        )
        wait_for_bolt(settings)
        return

    if action == "wait_for_index_online":
        print_action(f"Waiting for vector index {settings.index_name} to become ONLINE...")
        wait_for_bolt(settings)
        wait_for_index_online(settings)
        return

    raise ValueError(f"Unsupported setup action: {action}")


def ensure_setup(
    settings: BenchSettings,
    *,
    fresh: bool = False,
    reimport: bool = False,
    rebuild_runtime: bool = False,
    ensure_server_ready: bool = False,
) -> EnvironmentState:
    state = inspect_environment(settings)
    actions = compute_setup_actions(
        state,
        fresh=fresh,
        reimport=reimport,
        rebuild_runtime=rebuild_runtime,
        ensure_server=ensure_server_ready,
    )
    if not actions:
        print_action("Benchmark runtime is already ready.")
        return state

    setup_table = Table(box=None, show_header=True, header_style="bold cyan", pad_edge=False)
    setup_table.add_column("#", style="bold magenta", no_wrap=True)
    setup_table.add_column("Action", style="bold white")
    setup_table.add_column("What it does", style="white")
    for index, action in enumerate(actions, start=1):
        setup_table.add_row(str(index), action, SETUP_ACTION_DETAILS[action])
    console().print("")
    console().print(Panel(setup_table, title="Setup Plan", border_style="cyan", expand=False))
    console().print("")

    for action in actions:
        execute_setup_action(action, settings)
    return inspect_environment(settings)


def render_status(settings: BenchSettings, state: EnvironmentState) -> str:
    override_lines = (
        [f"  {key}: {value}" for key, value in sorted(settings.local_overrides.items())]
        if settings.local_overrides
        else ["  none"]
    )
    workspace_lines = format_kv_lines(
        [
            ("dataset", settings.dataset),
            ("runtime_root", str(settings.runtime_root)),
            ("runtime_home", str(settings.runtime_home)),
            ("runtime_archive", str(settings.runtime_archive)),
            ("uri", settings.uri),
            ("database", settings.database),
            ("results_root", str(settings.results_root)),
        ]
    )
    check_lines = format_kv_lines(
        [
            ("python_env", check_label(state.venv_ready)),
            ("dataset_artifacts", check_label(state.dataset_ready)),
            ("runtime_archive", check_label(state.runtime_archive_ready)),
            ("runtime_extracted", check_label(state.runtime_extracted)),
            ("runtime_prepared", check_label(state.runtime_prepared)),
            ("import_marker", state.import_state),
            ("neo4j_running", "yes" if state.server_running else "no"),
            ("bolt_ready", "yes" if state.bolt_ready else "no"),
            ("vector_index", state.index_state or "unknown"),
        ]
    )
    lines = [
        "Benchmark workspace status",
        "==========================",
        "",
        "Workspace",
        *workspace_lines,
        "",
        "Readiness",
        *check_lines,
        "",
        "Suggested next step",
        f"  {next_step_hint(state)}",
        "",
        f"Local overrides ({path_for_display(settings.local_overrides_path)})",
        *override_lines,
    ]
    return "\n".join(lines)


def print_status_view(settings: BenchSettings, state: EnvironmentState) -> None:
    workspace_table = Table(box=None, show_header=False, pad_edge=False)
    workspace_table.add_column(style="bold cyan", no_wrap=True)
    workspace_table.add_column(style="white")
    for label, value in (
        ("dataset", settings.dataset),
        ("runtime_root", str(settings.runtime_root)),
        ("runtime_home", str(settings.runtime_home)),
        ("runtime_archive", str(settings.runtime_archive)),
        ("uri", settings.uri),
        ("database", settings.database),
        ("results_root", str(settings.results_root)),
    ):
        workspace_table.add_row(label, value)

    readiness_table = Table(box=None, show_header=False, pad_edge=False)
    readiness_table.add_column(style="bold cyan", no_wrap=True)
    readiness_table.add_column()
    for label, value in (
        ("python_env", check_label(state.venv_ready)),
        ("dataset_artifacts", check_label(state.dataset_ready)),
        ("runtime_archive", check_label(state.runtime_archive_ready)),
        ("runtime_extracted", check_label(state.runtime_extracted)),
        ("runtime_prepared", check_label(state.runtime_prepared)),
        ("import_marker", state.import_state),
        ("neo4j_running", "yes" if state.server_running else "no"),
        ("bolt_ready", "yes" if state.bolt_ready else "no"),
        ("vector_index", state.index_state or "unknown"),
    ):
        readiness_table.add_row(label, Text(value, style=status_style(value)))

    override_table = Table(box=None, show_header=False, pad_edge=False)
    override_table.add_column(style="bold cyan", no_wrap=True)
    override_table.add_column(style="white")
    if settings.local_overrides:
        for key, value in sorted(settings.local_overrides.items()):
            override_table.add_row(key, str(value))
    else:
        override_table.add_row("overrides", Text("none", style="dim"))

    c = console()
    c.print(Panel(workspace_table, title="Workspace", border_style="cyan", expand=False))
    c.print(Panel(readiness_table, title="Readiness", border_style="blue", expand=False))
    c.print(Panel(next_step_hint(state), title="Next Step", border_style="green", expand=False))
    c.print(
        Panel(
            override_table,
            title=f"Local Overrides ({path_for_display(settings.local_overrides_path)})",
            border_style="magenta",
            expand=False,
        )
    )


def doctor_findings(settings: BenchSettings, state: EnvironmentState) -> list[str]:
    findings: list[str] = []
    if resolve_java_home(settings) is None:
        findings.append("JAVA_HOME could not be resolved. Set config/bench_local.json or install Java 17/21.")
    if not state.venv_ready:
        findings.append("Python environment is missing. Run ./scripts/bench.sh setup.")
    if not state.dataset_ready:
        findings.append("Dataset artifacts are missing. Run ./scripts/bench.sh setup.")
    if not state.runtime_archive_ready:
        findings.append("Packaged runtime archive is missing. Run ./scripts/bench.sh setup.")
    if not state.runtime_extracted:
        findings.append("Extracted runtime is missing. Run ./scripts/bench.sh setup.")
    if not state.runtime_prepared:
        findings.append("Runtime plugin/config preparation is incomplete. Run ./scripts/bench.sh setup.")
    if state.import_state == "missing":
        findings.append("Imported ogbn-arxiv database is missing. Run ./scripts/bench.sh setup.")
    if state.import_state == "mismatch":
        findings.append("Imported ogbn-arxiv fingerprint does not match the current processed manifest. Run ./scripts/bench.sh setup --reimport.")
    if state.server_running and not state.bolt_ready:
        findings.append("Neo4j reports running but Bolt is not ready yet.")
    if state.server_running and state.bolt_ready and not state.index_ready:
        findings.append(f"Vector index {settings.index_name} is not ONLINE.")
    return findings


def write_temp_neo4j_config(settings: BenchSettings) -> Path:
    payload = {
        "uri": settings.uri,
        "user": settings.user,
        "password": settings.password,
        "database": settings.database,
    }
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False)
    with handle:
        json.dump(payload, handle)
    return Path(handle.name)


def summaries_dir(settings: BenchSettings) -> Path:
    return settings.results_root / "summaries"


def summary_artifact_kind(payload: dict[str, Any]) -> str:
    if "native_rag" in payload and "two_call_baseline" in payload:
        return "comparison"
    if "method" in payload and "latency_ms" in payload:
        return "method"
    return "unknown"


def summary_artifact_files(settings: BenchSettings, *, kind: str = "all") -> list[Path]:
    directory = summaries_dir(settings)
    if not directory.exists():
        return []
    files = [path for path in directory.glob("*.json") if path.name != ".gitkeep"]
    filtered: list[Path] = []
    for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True):
        payload = load_json(path)
        artifact_kind = summary_artifact_kind(payload)
        if kind == "all" or artifact_kind == kind:
            filtered.append(path)
    return filtered


def resolve_summary_artifact(settings: BenchSettings, *, requested_path: str | None, kind: str) -> Path:
    if requested_path:
        path = resolve_path(requested_path, settings.benchmark_root)
        if not path.exists():
            raise RuntimeError(f"Results file does not exist: {path}")
        return path

    preferred_kind = "comparison" if kind == "all" else kind
    candidates = summary_artifact_files(settings, kind=preferred_kind)
    if not candidates and kind == "all":
        candidates = summary_artifact_files(settings, kind="method")
    if not candidates:
        raise RuntimeError(f"No {kind} result summaries were found under {summaries_dir(settings)}")
    return candidates[0]


def format_latency_value(latency: dict[str, Any], key: str) -> str:
    return f"{float(latency.get(key, 0.0)):.3f}"


def render_recent_results(settings: BenchSettings, *, limit: int, kind: str) -> None:
    artifacts = summary_artifact_files(settings, kind=kind)
    if not artifacts:
        console().print(
            Panel(
                f"No {kind} result summaries were found under {summaries_dir(settings)}",
                title="Results",
                border_style="yellow",
                expand=False,
            )
        )
        return

    table = Table(box=None, show_header=True, header_style="bold cyan", pad_edge=False)
    table.add_column("#", style="bold magenta", no_wrap=True)
    table.add_column("Type", style="bold white")
    table.add_column("Name", style="white")
    table.add_column("Modified", style="white")
    table.add_column("Path", style="dim")
    selected = artifacts[: max(limit, 1)]
    for index, path in enumerate(selected, start=1):
        payload = load_json(path)
        artifact_kind = summary_artifact_kind(payload)
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(
            str(index),
            artifact_kind,
            path.name,
            modified,
            path_for_display(path),
        )

    console().print(
        Panel(
            table,
            title=f"Recent Results ({len(selected)} shown)",
            subtitle=f"filter={kind}",
            border_style="cyan",
            expand=False,
        )
    )


def render_method_results(path: Path, payload: dict[str, Any]) -> None:
    latency = payload["latency_ms"]
    table = Table(box=None, show_header=True, header_style="bold cyan", pad_edge=False)
    table.add_column("Method", style="bold white")
    table.add_column("Count", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    table.add_row(
        str(payload.get("method", "unknown")),
        str(int(float(latency.get("count", 0.0)))),
        format_latency_value(latency, "median_ms"),
        format_latency_value(latency, "p95_ms"),
        format_latency_value(latency, "p99_ms"),
        format_latency_value(latency, "max_ms"),
    )
    details = Table(box=None, show_header=False, pad_edge=False)
    details.add_column(style="bold cyan", no_wrap=True)
    details.add_column(style="white")
    details.add_row("query_count", str(payload.get("query_count", "unknown")))
    details.add_row("warmup_runs", str(payload.get("warmup_runs", "unknown")))
    details.add_row("measured_runs", str(payload.get("measured_runs", "unknown")))
    details.add_row("raw_results", path_for_display(Path(payload.get("raw_results_path", ""))) if payload.get("raw_results_path") else "unknown")
    details.add_row("log_path", path_for_display(Path(payload.get("log_path", ""))) if payload.get("log_path") else "unknown")
    details.add_row("summary_path", path_for_display(path))
    c = console()
    c.print(Panel(table, title="Latest Method Result", border_style="green", expand=False))
    c.print(Panel(details, title="Artifacts", border_style="cyan", expand=False))


def render_comparison_results(path: Path, payload: dict[str, Any]) -> None:
    summary_table = Table(box=None, show_header=True, header_style="bold cyan", pad_edge=False)
    summary_table.add_column("Method", style="bold white")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Median (ms)", justify="right")
    summary_table.add_column("P95 (ms)", justify="right")
    summary_table.add_column("P99 (ms)", justify="right")
    summary_table.add_column("Max (ms)", justify="right")
    for label in ("native_rag", "two_call_baseline"):
        latency = payload[label]["latency_ms"]
        summary_table.add_row(
            label,
            str(int(float(latency.get("count", 0.0)))),
            format_latency_value(latency, "median_ms"),
            format_latency_value(latency, "p95_ms"),
            format_latency_value(latency, "p99_ms"),
            format_latency_value(latency, "max_ms"),
        )
    match_count = sum(1 for query in payload.get("queries", {}).values() if query.get("result_rows_match"))
    total_queries = len(payload.get("queries", {}))
    details = Table(box=None, show_header=False, pad_edge=False)
    details.add_column(style="bold cyan", no_wrap=True)
    details.add_column(style="white")
    details.add_row("order", str(payload.get("order", "unknown")))
    details.add_row("query_count", str(total_queries))
    details.add_row("result_row_matches", f"{match_count}/{total_queries}")
    details.add_row("summary_path", path_for_display(path))
    c = console()
    c.print(Panel(summary_table, title="Latest Comparison Result", border_style="green", expand=False))
    c.print(Panel(details, title="Comparison Details", border_style="cyan", expand=False))


def render_results(settings: BenchSettings, *, requested_path: str | None, recent: int, kind: str) -> None:
    if recent > 0:
        render_recent_results(settings, limit=recent, kind=kind)
        return

    path = resolve_summary_artifact(settings, requested_path=requested_path, kind=kind)
    payload = load_json(path)
    artifact_kind = summary_artifact_kind(payload)
    if artifact_kind == "comparison":
        render_comparison_results(path, payload)
        return
    if artifact_kind == "method":
        render_method_results(path, payload)
        return
    raise RuntimeError(f"Unsupported results summary format: {path}")


def parse_benchmark_paths(output_lines: list[str]) -> dict[str, list[Path] | Path]:
    summary_paths: list[Path] = []
    raw_paths: list[Path] = []
    comparison_path: Path | None = None
    for line in output_lines:
        if line.startswith("Wrote raw results to "):
            raw_paths.append(Path(line.removeprefix("Wrote raw results to ").strip()))
        elif line.startswith("Wrote summary to "):
            summary_paths.append(Path(line.removeprefix("Wrote summary to ").strip()))
        elif line.startswith("Wrote comparison summary to "):
            comparison_path = Path(line.removeprefix("Wrote comparison summary to ").strip())
    return {
        "raw_paths": raw_paths,
        "summary_paths": summary_paths,
        "comparison_path": comparison_path,
    }


def print_run_summary(mode: str, paths: dict[str, list[Path] | Path]) -> None:
    summary_paths = [path for path in paths["summary_paths"] if isinstance(path, Path)]
    comparison_path = paths["comparison_path"]
    if comparison_path and isinstance(comparison_path, Path) and comparison_path.exists():
        comparison = load_json(comparison_path)
        summary_table = Table(box=None, show_header=True, header_style="bold cyan", pad_edge=False)
        summary_table.add_column("Method", style="bold white")
        summary_table.add_column("Median (ms)", justify="right")
        summary_table.add_column("P95 (ms)", justify="right")
        summary_table.add_column("P99 (ms)", justify="right")
        for label in ("native_rag", "two_call_baseline"):
            latency = comparison[label]["latency_ms"]
            summary_table.add_row(
                label,
                f"{latency['median_ms']:.3f}",
                f"{latency['p95_ms']:.3f}",
                f"{latency['p99_ms']:.3f}",
            )
        matches = sum(1 for query in comparison["queries"].values() if query["result_rows_match"])
        total = len(comparison["queries"])
        c = console()
        c.print("")
        c.print(Panel(summary_table, title=f"Benchmark Summary ({mode})", border_style="green", expand=False))
        c.print(Panel(f"result row matches: {matches}/{total}\ncomparison summary: {comparison_path}", title="Outcome", border_style="cyan", expand=False))
        return

    if summary_paths:
        summary_table = Table(box=None, show_header=True, header_style="bold cyan", pad_edge=False)
        summary_table.add_column("Method", style="bold white")
        summary_table.add_column("Median (ms)", justify="right")
        summary_table.add_column("P95 (ms)", justify="right")
        summary_table.add_column("P99 (ms)", justify="right")
        for summary_path in summary_paths:
            summary = load_json(summary_path)
            latency = summary["latency_ms"]
            summary_table.add_row(
                summary["method"],
                f"{latency['median_ms']:.3f}",
                f"{latency['p95_ms']:.3f}",
                f"{latency['p99_ms']:.3f}",
            )
        c = console()
        c.print("")
        c.print(Panel(summary_table, title=f"Benchmark Summary ({mode})", border_style="green", expand=False))
        c.print(
            Panel(
                "\n".join(f"summary path: {summary_path}" for summary_path in summary_paths),
                title="Artifacts",
                border_style="cyan",
                expand=False,
            )
        )
        return

    console().print("")
    console().print(
        Panel(
            f"Benchmark finished in mode {mode}, but no summary paths were parsed from the runner output.",
            title="Benchmark Summary",
            border_style="yellow",
            expand=False,
        )
    )


def run_benchmark_command(
    settings: BenchSettings,
    *,
    tier: str,
    mode: str,
    order: str | None,
) -> None:
    mode_profile = settings.mode_profiles[mode]
    manifest_path = settings.tier_manifests[tier]
    resolved_order = order or mode_profile.default_order
    temp_config_path = write_temp_neo4j_config(settings)

    run_table = Table(box=None, show_header=False, pad_edge=False)
    run_table.add_column(style="bold cyan", no_wrap=True)
    run_table.add_column(style="white")
    for label, value in (
        ("tier", f"{tier} - {TIER_DESCRIPTIONS[tier]}"),
        ("mode", f"{mode} - {MODE_DESCRIPTIONS[mode]}"),
        ("manifest", str(manifest_path)),
        ("runtime_home", str(settings.runtime_home)),
        ("defaults", str(mode_profile.defaults_path)),
        ("results_root", str(settings.results_root)),
        ("order", resolved_order or "n/a"),
    ):
        run_table.add_row(label, value)
    c = console()
    c.print(Panel(run_table, title="Resolved Benchmark Run", border_style="cyan", expand=False))
    c.print("")

    command = [
        str(settings.benchmark_root / "scripts" / "run_benchmark.sh"),
        "--neo4j-config",
        str(temp_config_path),
        "--case-manifest",
        str(manifest_path),
        "--defaults",
        str(mode_profile.defaults_path),
        "--output-root",
        str(settings.results_root),
    ]
    if mode_profile.comparison:
        command.extend(
            [
                "--run-comparison",
                "--server-home",
                str(settings.runtime_home),
            ]
        )
        if resolved_order:
            command.extend(["--order", resolved_order])
    else:
        command.extend(["--method", str(mode_profile.method)])

    try:
        output_lines = stream_command(command, settings=settings, require_java=True)
    finally:
        temp_config_path.unlink(missing_ok=True)

    print_run_summary(mode, parse_benchmark_paths(output_lines))


def do_status(settings: BenchSettings) -> int:
    print_status_view(settings, inspect_environment(settings))
    return 0


def do_doctor(settings: BenchSettings) -> int:
    state = inspect_environment(settings)
    print_status_view(settings, state)
    findings = doctor_findings(settings, state)
    if not findings:
        console().print("")
        console().print(Panel("All checks passed.", title="Doctor", border_style="green", expand=False))
        return 0

    findings_table = Table(box=None, show_header=True, header_style="bold yellow", pad_edge=False)
    findings_table.add_column("#", style="bold magenta", no_wrap=True)
    findings_table.add_column("Finding", style="white")
    for index, finding in enumerate(findings, start=1):
        findings_table.add_row(str(index), finding)
    console().print("")
    console().print(Panel(findings_table, title="Doctor Findings", border_style="yellow", expand=False))
    return 1


def do_setup(settings: BenchSettings, args: argparse.Namespace) -> int:
    ensure_setup(
        settings,
        fresh=bool(args.fresh),
        reimport=bool(args.reimport),
        rebuild_runtime=bool(args.rebuild_runtime),
        ensure_server_ready=False,
    )
    console().print("")
    console().print(Panel("Setup complete.", title="Setup", border_style="green", expand=False))
    return 0


def do_start(settings: BenchSettings) -> int:
    ensure_setup(settings, ensure_server_ready=False)
    if server_running(settings):
        console().print(Panel(f"Neo4j runtime is already running at {settings.runtime_home}", title="Start", border_style="blue", expand=False))
        return 0

    execute_setup_action("start_server", settings)
    wait_for_index_online(settings)
    console().print(Panel(f"Neo4j runtime is running at {settings.runtime_home}", title="Start", border_style="green", expand=False))
    return 0


def do_stop(settings: BenchSettings) -> int:
    if not settings.runtime_home.exists():
        console().print(Panel(f"Runtime does not exist yet: {settings.runtime_home}", title="Stop", border_style="yellow", expand=False))
        return 1

    if not server_running(settings):
        console().print(Panel("Neo4j runtime is not running.", title="Stop", border_style="blue", expand=False))
        return 0

    stream_command(
        [str(settings.runtime_home / "bin" / "neo4j"), "stop"],
        settings=settings,
        require_java=True,
    )
    console().print(Panel("Neo4j runtime stopped.", title="Stop", border_style="green", expand=False))
    return 0


def do_results(settings: BenchSettings, args: argparse.Namespace) -> int:
    render_results(settings, requested_path=getattr(args, "file", None), recent=int(getattr(args, "recent", 0) or 0), kind=str(getattr(args, "kind", "all")))
    return 0


def do_run(settings: BenchSettings, args: argparse.Namespace) -> int:
    ensure_setup(
        settings,
        fresh=bool(args.fresh),
        reimport=bool(args.reimport),
        rebuild_runtime=bool(args.rebuild_runtime),
        ensure_server_ready=True,
    )
    run_benchmark_command(settings, tier=args.tier, mode=args.mode, order=args.order)
    return 0


def dispatch(parsed_args: argparse.Namespace) -> int:
    settings = resolve_settings(parsed_args)

    if parsed_args.command == "status":
        return do_status(settings)
    if parsed_args.command == "doctor":
        return do_doctor(settings)
    if parsed_args.command == "setup":
        return do_setup(settings, parsed_args)
    if parsed_args.command == "start":
        return do_start(settings)
    if parsed_args.command == "stop":
        return do_stop(settings)
    if parsed_args.command == "results":
        return do_results(settings, parsed_args)
    if parsed_args.command == "run":
        return do_run(settings, parsed_args)
    raise AssertionError(f"Unhandled command: {parsed_args.command}")


def do_shell(shell_args: argparse.Namespace) -> int:
    shell_settings = resolve_settings(shell_args)
    interactive_tty = sys.stdin.isatty() and sys.stdout.isatty()
    session = build_prompt_session() if interactive_tty else None
    console().print(Panel(render_shell_banner(shell_settings), border_style="cyan", expand=False))
    while True:
        try:
            if session is not None:
                line = session.prompt(prompt_message(shell_settings), bottom_toolbar=prompt_toolbar())
            else:
                line = input(shell_prompt(shell_settings))
        except EOFError:
            console().print("")
            return 0
        except KeyboardInterrupt:
            console().print("")
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {"exit", "quit"}:
            return 0
        if stripped == "help":
            console().print(Panel(render_shell_help(), title="Help", border_style="cyan", expand=False))
            continue
        if stripped.startswith("help "):
            console().print(
                Panel(render_shell_help(stripped.split(maxsplit=1)[1]), title="Help", border_style="cyan", expand=False)
            )
            continue

        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            console(stderr=True).print(Panel(f"Parse error: {exc}", title="Input Error", border_style="red", expand=False))
            continue

        if not tokens:
            continue
        if tokens[0] == "shell":
            console(stderr=True).print(Panel("Already inside the benchmark shell.", title="Shell", border_style="yellow", expand=False))
            continue
        if tokens[0] not in SHELL_COMMANDS:
            console(stderr=True).print(
                Panel(
                    f"Unknown command: {tokens[0]}. Type `help` for available commands.",
                    title="Shell",
                    border_style="red",
                    expand=False,
                )
            )
            continue

        try:
            exit_code = dispatch(parse_args(shell_command_argv(shell_args, tokens)))
        except SystemExit as exc:
            exit_code = int(exc.code) if isinstance(exc.code, int) else 1
        except subprocess.CalledProcessError as exc:
            command_text = " ".join(shlex.quote(part) for part in exc.cmd)
            console(stderr=True).print(
                Panel(
                    f"Command failed with exit code {exc.returncode}:\n{command_text}",
                    title="Command Failed",
                    border_style="red",
                    expand=False,
                )
            )
            exit_code = exc.returncode or 1
        except RuntimeError as exc:
            console(stderr=True).print(Panel(str(exc), title="Runtime Error", border_style="red", expand=False))
            exit_code = 1
        except KeyboardInterrupt:
            console(stderr=True).print(Panel("Interrupted.", title="Shell", border_style="yellow", expand=False))
            exit_code = 130

        if exit_code not in {0, None}:
            console(stderr=True).print(
                Panel(f"Command exited with status {exit_code}", title="Shell", border_style="yellow", expand=False)
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "shell":
        return do_shell(args)

    try:
        return dispatch(args)
    except subprocess.CalledProcessError as exc:
        command_text = " ".join(shlex.quote(part) for part in exc.cmd)
        console(stderr=True).print(
            Panel(
                f"Command failed with exit code {exc.returncode}:\n{command_text}",
                title="Command Failed",
                border_style="red",
                expand=False,
            )
        )
        return exc.returncode or 1
    except RuntimeError as exc:
        console(stderr=True).print(Panel(str(exc), title="Runtime Error", border_style="red", expand=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
