from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.upload_training_log_to_wandb import _discover_local_wandb_run, upload_training_log


class _FakeRemoteRun:
    def __init__(self, summary: dict[str, object]) -> None:
        self.summary = summary


class _FakeApi:
    def __init__(self, remote_summary: dict[str, object] | None, should_raise: bool = False) -> None:
        self.remote_summary = remote_summary
        self.should_raise = should_raise
        self.requested_paths: list[str] = []

    def run(self, path: str) -> _FakeRemoteRun:
        self.requested_paths.append(path)
        if self.should_raise:
            raise RuntimeError("remote lookup failed")
        if self.remote_summary is None:
            raise RuntimeError("run not found")
        return _FakeRemoteRun(self.remote_summary)


class _FakeRun:
    def __init__(self, entity: str | None, project: str, run_id: str | None) -> None:
        self.entity = entity
        self.project = project
        self.id = run_id
        self.summary: dict[str, object] = {}


class _FakeWandb:
    def __init__(self, remote_summary: dict[str, object] | None = None, api_should_raise: bool = False) -> None:
        self.remote_summary = remote_summary
        self.api_should_raise = api_should_raise
        self.init_calls: list[dict[str, object]] = []
        self.logged_rows: list[tuple[dict[str, object], int | None]] = []
        self.finished = False
        self.api_instance = _FakeApi(remote_summary, should_raise=api_should_raise)
        self.run: _FakeRun | None = None

    def init(self, **kwargs) -> _FakeRun:
        self.init_calls.append(kwargs)
        self.run = _FakeRun(
            entity=kwargs.get("entity"),
            project=str(kwargs["project"]),
            run_id=kwargs.get("id"),
        )
        return self.run

    def Api(self) -> _FakeApi:
        return self.api_instance

    def log(self, row: dict[str, object], step: int | None = None) -> None:
        self.logged_rows.append((row, step))

    def finish(self) -> None:
        self.finished = True


class UploadTrainingLogToWandbTests(unittest.TestCase):
    def _write_training_log(self, run_dir: Path) -> Path:
        log_path = run_dir / "training_metrics.jsonl"
        events = [
            {
                "event": "train",
                "step": 10,
                "epoch": 1,
                "loss_nats_per_token": 1.5,
                "bits_per_base": 0.7,
                "learning_rate": 1e-3,
                "tokens_per_second": 123.0,
            },
            {
                "event": "eval",
                "split": "val",
                "step": 20,
                "epoch": 1,
                "loss_nats_per_token": 1.2,
                "bits_per_base": 0.6,
            },
            {
                "event": "train",
                "step": 30,
                "epoch": 1,
                "loss_nats_per_token": 1.1,
                "bits_per_base": 0.5,
                "learning_rate": 5e-4,
                "tokens_per_second": 234.0,
            },
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event) + "\n")
        return log_path

    def _make_args(self, run_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            run_dir=str(run_dir),
            log_file=None,
            project="dna-compress",
            entity="test-entity",
            name=None,
            mode="online",
            resolved_config=None,
            job_type="train",
        )

    def _create_wandb_run_dir(self, run_dir: Path, dirname: str, run_id: str) -> Path:
        local_run_dir = run_dir / "wandb" / dirname
        local_run_dir.mkdir(parents=True, exist_ok=True)
        (local_run_dir / f"run-{run_id}.wandb").write_text("", encoding="utf-8")
        return local_run_dir

    def test_discover_local_wandb_run_prefers_latest_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            older = self._create_wandb_run_dir(run_dir, "run-20260412_224252-older01", "older01")
            newer = self._create_wandb_run_dir(run_dir, "run-20260412_224253-newer02", "newer02")
            latest_run = run_dir / "wandb" / "latest-run"
            latest_run.symlink_to(older, target_is_directory=True)

            discovered = _discover_local_wandb_run(run_dir)

            self.assertEqual(discovered, (older, "older01"))
            self.assertNotEqual(discovered, (newer, "newer02"))

    def test_upload_training_log_resumes_only_missing_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_training_log(run_dir)
            self._create_wandb_run_dir(run_dir, "run-20260412_224252-resume01", "resume01")
            args = self._make_args(run_dir)
            fake_wandb = _FakeWandb(remote_summary={"uploaded_events": 2})

            message = upload_training_log(args, fake_wandb)

            self.assertIn("resumed_from_event=2", message)
            self.assertEqual(len(fake_wandb.logged_rows), 1)
            self.assertEqual(fake_wandb.logged_rows[0][1], 30)
            self.assertEqual(fake_wandb.init_calls[0]["id"], "resume01")
            self.assertEqual(fake_wandb.init_calls[0]["resume"], "allow")
            self.assertEqual(fake_wandb.api_instance.requested_paths, ["test-entity/dna-compress/resume01"])
            self.assertTrue(fake_wandb.finished)
            assert fake_wandb.run is not None
            self.assertEqual(fake_wandb.run.summary["uploaded_events"], 3)
            self.assertEqual(fake_wandb.run.summary["upload_resumed_run_id"], "resume01")
            self.assertEqual(fake_wandb.run.summary["upload_resumed_from_event"], 2)
            self.assertEqual(fake_wandb.run.summary["upload_remote_uploaded_events_before_resume"], 2)
            self.assertEqual(fake_wandb.run.summary["best_val/bpb"], 0.6)

    def test_upload_training_log_replays_full_log_when_remote_progress_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_training_log(run_dir)
            self._create_wandb_run_dir(run_dir, "run-20260412_224252-resume01", "resume01")
            args = self._make_args(run_dir)
            fake_wandb = _FakeWandb(remote_summary={})

            message = upload_training_log(args, fake_wandb)

            self.assertIn("resumed_from_event=0", message)
            self.assertEqual([step for _, step in fake_wandb.logged_rows], [10, 20, 30])
            assert fake_wandb.run is not None
            self.assertEqual(fake_wandb.run.summary["uploaded_events"], 3)
            self.assertEqual(fake_wandb.run.summary["upload_resumed_run_id"], "resume01")
            self.assertEqual(fake_wandb.run.summary["upload_resumed_from_event"], 0)
            self.assertNotIn("upload_remote_uploaded_events_before_resume", fake_wandb.run.summary)

    def test_upload_training_log_falls_back_when_local_wandb_state_is_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_training_log(run_dir)
            broken_run_dir = run_dir / "wandb" / "run-20260412_224252-broken01"
            broken_run_dir.mkdir(parents=True, exist_ok=True)
            args = self._make_args(run_dir)
            fake_wandb = _FakeWandb(api_should_raise=True)

            message = upload_training_log(args, fake_wandb)

            self.assertIn("resumed_from_event=0", message)
            self.assertEqual(len(fake_wandb.logged_rows), 3)
            self.assertNotIn("id", fake_wandb.init_calls[0])
            self.assertEqual(fake_wandb.api_instance.requested_paths, [])
            assert fake_wandb.run is not None
            self.assertEqual(fake_wandb.run.summary["uploaded_events"], 3)
            self.assertNotIn("upload_resumed_run_id", fake_wandb.run.summary)


if __name__ == "__main__":
    unittest.main()
