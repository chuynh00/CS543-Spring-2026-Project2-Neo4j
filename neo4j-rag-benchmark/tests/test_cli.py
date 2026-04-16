from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.cli import (
    BenchSettings,
    EnvironmentState,
    compute_setup_actions,
    doctor_findings,
    main,
    parse_args,
    render_results,
    render_shell_help,
    render_status,
    resolve_settings,
    run_benchmark_command,
    summary_artifact_files,
    shell_command_argv,
)


class BenchmarkCliTest(unittest.TestCase):
    def _write_profiles(self, root: Path) -> tuple[Path, Path]:
        config_dir = root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        profiles_path = config_dir / "bench_profiles.json"
        local_path = config_dir / "bench_local.json"
        profiles_path.write_text(
            json.dumps(
                {
                    "defaults": {
                        "dataset": "ogbn-arxiv",
                        "runtime_version": "5.26.0",
                        "runtime_root": "../neo4j-rag-runtime",
                        "uri": "bolt://127.0.0.1:7687",
                        "database": "neo4j",
                        "results_root": "results",
                        "index_name": "paper_embedding_idx",
                        "processed_manifest": "data/processed/ogbn-arxiv/manifest.json",
                        "processed_dir": "data/processed/ogbn-arxiv",
                    },
                    "tiers": {
                        "smoke": {"query_manifest": "data/query_sets/ogbn_arxiv/smoke_manifest.json"},
                        "dev": {"query_manifest": "data/query_sets/ogbn_arxiv/dev_manifest.json"},
                        "full": {"query_manifest": "data/query_sets/ogbn_arxiv/full_manifest.json"},
                    },
                    "modes": {
                        "rerank": {
                            "comparison": True,
                            "defaults": "config/benchmark_defaults.json",
                            "order": "native-first",
                        },
                        "parity": {
                            "comparison": True,
                            "defaults": "config/benchmark_defaults_parity.json",
                            "order": "native-first",
                        },
                        "native-only": {
                            "comparison": False,
                            "method": "native",
                            "defaults": "config/benchmark_defaults.json",
                        },
                        "baseline-only": {
                            "comparison": False,
                            "method": "baseline",
                            "defaults": "config/benchmark_defaults.json",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return profiles_path, local_path

    def test_parse_args_supports_run_mode(self) -> None:
        args = parse_args(["run", "--tier", "smoke", "--mode", "rerank"])

        self.assertEqual(args.command, "run")
        self.assertEqual(args.tier, "smoke")
        self.assertEqual(args.mode, "rerank")

    def test_parse_args_supports_results_mode(self) -> None:
        args = parse_args(["results", "--recent", "3", "--kind", "comparison"])

        self.assertEqual(args.command, "results")
        self.assertEqual(args.recent, 3)
        self.assertEqual(args.kind, "comparison")

    def test_parse_args_defaults_to_shell_when_no_command_is_provided(self) -> None:
        args = parse_args([])

        self.assertEqual(args.command, "shell")

    def test_shell_command_argv_applies_shell_overrides(self) -> None:
        shell_args = parse_args(["shell", "--runtime-root", "/runtime-root", "--database", "benchdb"])

        argv = shell_command_argv(shell_args, ["status"])

        self.assertEqual(argv, ["status", "--runtime-root", "/runtime-root", "--database", "benchdb"])

    def test_render_shell_help_for_run_includes_modes_and_tiers(self) -> None:
        rendered = render_shell_help("run")

        self.assertIn("Command: run", rendered)
        self.assertIn("rerank", rendered)
        self.assertIn("smoke", rendered)

    def test_resolve_settings_prefers_cli_over_local_over_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_path, local_path = self._write_profiles(root)
            local_path.write_text(
                json.dumps(
                    {
                        "runtime_root": "/local/runtime-root",
                        "uri": "bolt://local:7687",
                        "database": "localdb",
                        "results_root": "/local/results",
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "status",
                    "--runtime-root",
                    "/cli/runtime-root",
                    "--uri",
                    "bolt://cli:7687",
                ]
            )

            settings = resolve_settings(
                args,
                benchmark_root=root,
                profiles_path=profiles_path,
                local_overrides_path=local_path,
            )

            self.assertEqual(settings.runtime_root, Path("/cli/runtime-root"))
            self.assertEqual(settings.uri, "bolt://cli:7687")
            self.assertEqual(settings.database, "localdb")
            self.assertEqual(settings.results_root, Path("/local/results"))

    def test_compute_setup_actions_for_missing_state(self) -> None:
        state = EnvironmentState(
            venv_ready=False,
            dataset_ready=False,
            runtime_archive_ready=False,
            runtime_extracted=False,
            runtime_prepared=False,
            import_state="missing",
            server_running=False,
            bolt_ready=False,
            index_state=None,
            index_ready=False,
            local_overrides_present=False,
        )

        actions = compute_setup_actions(
            state,
            fresh=False,
            reimport=False,
            rebuild_runtime=False,
            ensure_server=True,
        )

        self.assertEqual(
            actions,
            [
                "bootstrap_venv",
                "prepare_dataset",
                "build_runtime_archive",
                "extract_runtime",
                "prepare_runtime",
                "import_dataset",
            ],
        )

    def test_compute_setup_actions_rejects_import_mismatch(self) -> None:
        state = EnvironmentState(
            venv_ready=True,
            dataset_ready=True,
            runtime_archive_ready=True,
            runtime_extracted=True,
            runtime_prepared=True,
            import_state="mismatch",
            server_running=False,
            bolt_ready=False,
            index_state=None,
            index_ready=False,
            local_overrides_present=False,
        )

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            compute_setup_actions(
                state,
                fresh=False,
                reimport=False,
                rebuild_runtime=False,
                ensure_server=False,
            )

    def test_compute_setup_actions_starts_existing_runtime_when_needed(self) -> None:
        state = EnvironmentState(
            venv_ready=True,
            dataset_ready=True,
            runtime_archive_ready=True,
            runtime_extracted=True,
            runtime_prepared=True,
            import_state="ready",
            server_running=False,
            bolt_ready=False,
            index_state=None,
            index_ready=False,
            local_overrides_present=False,
        )

        actions = compute_setup_actions(
            state,
            fresh=False,
            reimport=False,
            rebuild_runtime=False,
            ensure_server=True,
        )

        self.assertEqual(actions, ["start_server", "wait_for_index_online"])

    def test_doctor_findings_report_missing_java_and_import_mismatch(self) -> None:
        settings = BenchSettings(
            benchmark_root=Path("/workspace/neo4j-rag-benchmark"),
            source_repo=Path("/workspace"),
            project_root=Path("/"),
            profiles_path=Path("/workspace/neo4j-rag-benchmark/config/bench_profiles.json"),
            local_overrides_path=Path("/workspace/neo4j-rag-benchmark/config/bench_local.json"),
            local_overrides={},
            dataset="ogbn-arxiv",
            runtime_version="5.26.0",
            runtime_root=Path("/runtime-root"),
            runtime_home=Path("/runtime-root/neo4j-community-5.26.0-native-rag"),
            runtime_archive=Path("/workspace/packaging/standalone/target/neo4j-community-5.26.0-native-rag.tar.gz"),
            uri="bolt://127.0.0.1:7687",
            database="neo4j",
            user="",
            password="",
            results_root=Path("/workspace/neo4j-rag-benchmark/results"),
            index_name="paper_embedding_idx",
            processed_manifest_path=Path("/workspace/neo4j-rag-benchmark/data/processed/ogbn-arxiv/manifest.json"),
            processed_dir=Path("/workspace/neo4j-rag-benchmark/data/processed/ogbn-arxiv"),
            tier_manifests={},
            mode_profiles={},
            java_home_override=None,
        )
        state = EnvironmentState(
            venv_ready=False,
            dataset_ready=False,
            runtime_archive_ready=False,
            runtime_extracted=False,
            runtime_prepared=False,
            import_state="mismatch",
            server_running=False,
            bolt_ready=False,
            index_state=None,
            index_ready=False,
            local_overrides_present=False,
        )

        with patch("benchmark.cli.resolve_java_home", return_value=None):
            findings = doctor_findings(settings, state)

        self.assertTrue(any("JAVA_HOME" in finding for finding in findings))
        self.assertTrue(any("setup --reimport" in finding for finding in findings))

    def test_render_status_includes_local_overrides(self) -> None:
        settings = BenchSettings(
            benchmark_root=Path("/workspace/neo4j-rag-benchmark"),
            source_repo=Path("/workspace"),
            project_root=Path("/"),
            profiles_path=Path("/workspace/neo4j-rag-benchmark/config/bench_profiles.json"),
            local_overrides_path=Path("/workspace/neo4j-rag-benchmark/config/bench_local.json"),
            local_overrides={"runtime_root": "/runtime-root"},
            dataset="ogbn-arxiv",
            runtime_version="5.26.0",
            runtime_root=Path("/runtime-root"),
            runtime_home=Path("/runtime-root/neo4j-community-5.26.0-native-rag"),
            runtime_archive=Path("/workspace/packaging/standalone/target/neo4j-community-5.26.0-native-rag.tar.gz"),
            uri="bolt://127.0.0.1:7687",
            database="neo4j",
            user="",
            password="",
            results_root=Path("/workspace/neo4j-rag-benchmark/results"),
            index_name="paper_embedding_idx",
            processed_manifest_path=Path("/workspace/neo4j-rag-benchmark/data/processed/ogbn-arxiv/manifest.json"),
            processed_dir=Path("/workspace/neo4j-rag-benchmark/data/processed/ogbn-arxiv"),
            tier_manifests={},
            mode_profiles={},
            java_home_override=None,
        )
        state = EnvironmentState(
            venv_ready=True,
            dataset_ready=True,
            runtime_archive_ready=True,
            runtime_extracted=True,
            runtime_prepared=True,
            import_state="ready",
            server_running=True,
            bolt_ready=True,
            index_state="ONLINE",
            index_ready=True,
            local_overrides_present=True,
        )

        rendered = render_status(settings, state)

        self.assertIn("runtime_root: /runtime-root", rendered)
        self.assertIn("vector_index      : ONLINE", rendered)

    def test_run_benchmark_command_builds_expected_comparison_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_path, local_path = self._write_profiles(root)
            (root / "scripts").mkdir(parents=True, exist_ok=True)
            (root / "scripts" / "run_benchmark.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            args = parse_args(["run", "--tier", "smoke", "--mode", "rerank"])
            settings = resolve_settings(
                args,
                benchmark_root=root,
                profiles_path=profiles_path,
                local_overrides_path=local_path,
            )
            captured = io.StringIO()

            with patch("benchmark.cli.stream_command") as stream_command_mock, patch(
                "benchmark.cli.print_run_summary"
            ), patch("sys.stdout", new=captured):
                stream_command_mock.return_value = [
                    "Wrote summary to /tmp/native.json",
                    "Wrote comparison summary to /tmp/comparison.json",
                ]
                run_benchmark_command(settings, tier="smoke", mode="rerank", order=None)

            command = stream_command_mock.call_args.args[0]
            self.assertIn("--run-comparison", command)
            self.assertIn("--server-home", command)
            self.assertIn(str(settings.runtime_home), command)
            self.assertIn(str(settings.tier_manifests["smoke"]), command)
            self.assertIn(str(settings.mode_profiles["rerank"].defaults_path), command)

    def test_summary_artifact_files_prefers_newest_and_filters_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_path, local_path = self._write_profiles(root)
            summaries = root / "results" / "summaries"
            summaries.mkdir(parents=True, exist_ok=True)
            method = summaries / "native_rag-1.json"
            comparison = summaries / "comparison-2.json"
            method.write_text(json.dumps({"method": "native_rag", "latency_ms": {"count": 1}}), encoding="utf-8")
            comparison.write_text(
                json.dumps({"native_rag": {"latency_ms": {"count": 1}}, "two_call_baseline": {"latency_ms": {"count": 1}}, "queries": {}}),
                encoding="utf-8",
            )
            args = parse_args(["results"])
            settings = resolve_settings(args, benchmark_root=root, profiles_path=profiles_path, local_overrides_path=local_path)

            all_files = summary_artifact_files(settings, kind="all")
            comparison_files = summary_artifact_files(settings, kind="comparison")

            self.assertEqual(all_files[0].name, "comparison-2.json")
            self.assertEqual([path.name for path in comparison_files], ["comparison-2.json"])

    def test_render_results_shows_latest_comparison_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles_path, local_path = self._write_profiles(root)
            summaries = root / "results" / "summaries"
            summaries.mkdir(parents=True, exist_ok=True)
            (summaries / "native_rag-1.json").write_text(
                json.dumps({"method": "native_rag", "latency_ms": {"count": 1, "median_ms": 1, "p95_ms": 1, "p99_ms": 1, "max_ms": 1}}),
                encoding="utf-8",
            )
            (summaries / "comparison-2.json").write_text(
                json.dumps(
                    {
                        "order": "native-first",
                        "native_rag": {"latency_ms": {"count": 1, "median_ms": 1, "p95_ms": 2, "p99_ms": 3, "max_ms": 4}},
                        "two_call_baseline": {"latency_ms": {"count": 1, "median_ms": 2, "p95_ms": 3, "p99_ms": 4, "max_ms": 5}},
                        "queries": {"q1": {"result_rows_match": True}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(["results"])
            settings = resolve_settings(args, benchmark_root=root, profiles_path=profiles_path, local_overrides_path=local_path)
            captured = io.StringIO()

            with patch("sys.stdout", new=captured):
                render_results(settings, requested_path=None, recent=0, kind="all")

            rendered = captured.getvalue()
            self.assertIn("Latest Comparison Result", rendered)
            self.assertIn("two_call_baseline", rendered)
            self.assertIn("result_row_matches", rendered)

    def test_main_shell_help_exits_cleanly(self) -> None:
        captured = io.StringIO()
        with patch("builtins.input", side_effect=["help", "exit"]), patch("sys.stdout", new=captured):
            exit_code = main([])

        rendered = captured.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Neo4j RAG Benchmark Shell", rendered)
        self.assertIn("Core commands", rendered)


if __name__ == "__main__":
    unittest.main()
