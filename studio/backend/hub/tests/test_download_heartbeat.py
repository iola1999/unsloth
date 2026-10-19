# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib
import json
import logging
import os
import time
from contextlib import contextmanager
from io import BytesIO
from types import SimpleNamespace

import pytest

from hub.services import download_lifecycle
from hub.services import snapshot_progress
from hub.utils import download_heartbeat, download_registry
from hub.workers import hf_download

VERDICT = "Download appears stalled (xet transport) -- no progress for 30s"


def _write_heartbeat(path, total):
    path.write_text(json.dumps({"activity_bytes": total}))


class _Proc:
    pid = 4242

    def __init__(
        self,
        heartbeat,
        rc = None,
    ):
        self.args = ["python", "--heartbeat", str(heartbeat)]
        self.killed = False
        self.rc = rc

    def poll(self):
        return self.rc

    def kill(self):
        self.killed = True


class _Stop:
    def __init__(self):
        self.stopped = False

    def set(self):
        self.stopped = True


def _watchdog(monkeypatch, proc, on_stall):
    import utils.hf_xet_fallback as shim

    started = []

    def fake_start(*, on_stall, **_kwargs):
        stop = _Stop()
        started.append((on_stall, stop))
        return stop

    monkeypatch.setattr(shim, "start_watchdog", fake_start)
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model")
    handle = download_lifecycle._start_stall_watchdog(
        registry,
        key,
        proc,
        repo_type = "model",
        repo_id = "Org/Model",
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        on_stall = on_stall,
    )
    return handle, started


def test_a_verdict_is_overruled_while_the_worker_still_receives(monkeypatch, tmp_path):
    beat = tmp_path / "beat"
    _write_heartbeat(beat, 100)
    proc = _Proc(beat)
    stalled = []
    _handle, started = _watchdog(monkeypatch, proc, stalled.append)
    _write_heartbeat(beat, 250)
    started[0][0](VERDICT)
    assert not proc.killed
    assert stalled == []
    assert len(started) == 2
    started[1][0](VERDICT)
    assert proc.killed
    assert stalled == [VERDICT]
    assert len(started) == 2


def test_a_heartbeat_older_than_the_stall_window_does_not_save_the_worker(monkeypatch, tmp_path):
    beat = tmp_path / "beat"
    _write_heartbeat(beat, 100)
    proc = _Proc(beat)
    stalled = []
    _handle, started = _watchdog(monkeypatch, proc, stalled.append)
    _write_heartbeat(beat, 250)
    stale = time.time() - 600
    os.utime(beat, (stale, stale))
    started[0][0](VERDICT)
    assert proc.killed
    assert stalled == [VERDICT]
    assert len(started) == 1


def test_a_failed_re_arm_lets_the_verdict_stand(monkeypatch, tmp_path):
    import utils.hf_xet_fallback as shim

    beat = tmp_path / "beat"
    _write_heartbeat(beat, 100)
    proc = _Proc(beat)
    stalled = []
    calls = []

    def fake_start(*, on_stall, **_kwargs):
        calls.append(on_stall)
        if len(calls) > 1:
            raise RuntimeError("no watchdog")
        return _Stop()

    monkeypatch.setattr(shim, "start_watchdog", fake_start)
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model")
    download_lifecycle._start_stall_watchdog(
        registry,
        key,
        proc,
        repo_type = "model",
        repo_id = "Org/Model",
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        on_stall = stalled.append,
    )
    _write_heartbeat(beat, 250)
    calls[0](VERDICT)
    assert proc.killed
    assert stalled == [VERDICT]


def test_a_verdict_with_no_heartbeat_kills_as_before(monkeypatch, tmp_path):
    proc = _Proc(tmp_path / "missing")
    stalled = []
    _handle, started = _watchdog(monkeypatch, proc, stalled.append)
    started[0][0](VERDICT)
    assert proc.killed
    assert stalled == [VERDICT]
    assert len(started) == 1


def test_stopping_the_handle_stops_the_re_armed_watchdog(monkeypatch, tmp_path):
    beat = tmp_path / "beat"
    _write_heartbeat(beat, 1)
    proc = _Proc(beat)
    handle, started = _watchdog(monkeypatch, proc, lambda _m: None)
    _write_heartbeat(beat, 2)
    started[0][0](VERDICT)
    handle.set()
    assert started[1][1].stopped
    _write_heartbeat(beat, 3)
    started[1][0](VERDICT)
    assert not proc.killed
    assert started[2][1].stopped


def test_the_worker_progress_class_reports_transfer_bytes(tmp_path):
    beat = tmp_path / "beat"
    cls = hf_download._progress_class(str(beat))
    bar = cls(total = 10, disable = True)
    bar.update(4)
    bar.update(-3)
    bar.close()
    assert download_heartbeat.read(str(beat)) == 4
    assert hf_download._progress_class(None) is None


def test_the_stall_window_follows_the_environment(monkeypatch):
    monkeypatch.delenv("UNSLOTH_XET_STALL_TIMEOUT", raising = False)
    assert download_lifecycle._stall_window() == 30.0
    monkeypatch.setenv("UNSLOTH_XET_STALL_TIMEOUT", "12")
    assert download_lifecycle._stall_window() == 12.0
    monkeypatch.setenv("UNSLOTH_XET_STALL_TIMEOUT", "junk")
    assert download_lifecycle._stall_window() == 30.0


def test_heartbeat_writer_throttles_and_reader_rejects_garbage(tmp_path):
    beat = tmp_path / "beat"
    writer = download_heartbeat.HeartbeatWriter(str(beat), interval = 3600.0)
    writer.add(5)
    writer.add(7)
    assert download_heartbeat.read(str(beat)) == 5
    beat.write_text("not a number")
    assert download_heartbeat.read(str(beat)) is None
    assert download_heartbeat.read(None) is None


def test_worker_files_are_removed_once_the_worker_exits(tmp_path):
    beat = tmp_path / "beat"
    _write_heartbeat(beat, 1)
    (tmp_path / "beat.tmp").write_text("")
    proc = _Proc(beat)
    download_lifecycle._cleanup_worker_files(proc)
    assert beat.exists()
    proc.rc = 0
    download_lifecycle._cleanup_worker_files(proc)
    assert not beat.exists()
    assert not (tmp_path / "beat.tmp").exists()


def test_an_orphaned_worker_removes_its_own_heartbeat(monkeypatch, tmp_path):
    beat = tmp_path / "beat"
    _write_heartbeat(beat, 1)
    (tmp_path / "beat.tmp").write_text("")

    def _exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(hf_download.os, "_exit", _exit)
    try:
        hf_download._terminate_orphaned_self(str(beat))
    except SystemExit as exc:
        assert exc.code == 130
    assert not beat.exists()
    assert not (tmp_path / "beat.tmp").exists()


def test_http_and_xet_workers_get_a_heartbeat(monkeypatch, tmp_path):
    spawned = []
    real_popen = download_lifecycle.subprocess.Popen

    def fake_popen(args, **kwargs):
        if "hub.workers.hf_download" not in args:
            return real_popen(args, **kwargs)
        spawned.append(args)
        return _Proc(tmp_path / "unused", rc = 0)

    monkeypatch.setattr(download_lifecycle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr("huggingface_hub.utils.get_token_to_send", lambda token: None)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    download_lifecycle.spawn_worker(
        ["--repo-id", "Org/Model"], None, use_xet = True, allow_ambient_token = False
    )
    download_lifecycle.spawn_worker(
        ["--repo-id", "Org/Model"], None, use_xet = False, allow_ambient_token = False
    )
    assert "--heartbeat" in spawned[0]
    assert spawned[0][spawned[0].index("--heartbeat") + 1].startswith(str(tmp_path))
    assert "--heartbeat" in spawned[1]


def test_network_counter_excludes_reconstruction_and_file_counts(monkeypatch, tmp_path):
    clock = [100.0]
    monkeypatch.setattr(download_heartbeat.time, "monotonic", lambda: clock[0])
    beat = tmp_path / "beat"
    cls = hf_download._progress_class(str(beat))
    network = cls(name = "huggingface_hub.snapshot_download.transfer", unit = "B", disable = True)
    reconstructed = cls(name = "huggingface_hub.snapshot_download", unit = "B", disable = True)
    files = cls(total = 4, disable = True)
    network.update(70_000_000)
    clock[0] += 2
    reconstructed.update(200_000_000)
    clock[0] += 2
    files.update(1)
    reading = download_heartbeat.read_transfer(str(beat))
    assert reading["network_bytes"] == 70_000_000
    assert reading["elapsed_seconds"] == 0
    assert download_heartbeat.read(str(beat)) == 270_000_001
    clock[0] += 2
    network.update(140_000_000)
    assert download_heartbeat.read_transfer(str(beat))["elapsed_seconds"] == 6
    assert download_heartbeat.read_transfer(str(beat))["network_bytes"] == 210_000_000
    for bar in (network, reconstructed, files):
        bar.close()


@pytest.mark.parametrize("transport", ["http", "xet"])
def test_hf_snapshot_callbacks_reach_the_network_counter(monkeypatch, tmp_path, transport):
    snapshot = importlib.import_module("huggingface_hub._snapshot_download")
    clock = [100.0]
    monkeypatch.setattr(download_heartbeat.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(snapshot, "read_tree_cache", lambda *args: {"weights.gguf": object()})
    writer_class = download_heartbeat.HeartbeatWriter
    monkeypatch.setattr(
        download_heartbeat, "HeartbeatWriter", lambda path: writer_class(path, interval = 0)
    )

    if transport == "http":
        import httpx
        from huggingface_hub import file_download

        @contextmanager
        def stream(**kwargs):
            response = httpx.Response(
                206,
                content = b"x" * 280,
                headers = {"Content-Length": "280", "Content-Range": "bytes 100-379/380"},
                request = httpx.Request("GET", kwargs["url"]),
            )
            iter_bytes = response.iter_bytes

            def chunks(**kwargs):
                for chunk in iter_bytes(**kwargs):
                    clock[0] += 1
                    yield chunk

            response.iter_bytes = chunks
            yield response

        monkeypatch.setattr(file_download, "http_stream_backoff", stream)
        monkeypatch.setattr(file_download.constants, "DOWNLOAD_CHUNK_SIZE", 70)

    def download(*args, tqdm_class, **kwargs):
        if transport == "http":
            output = BytesIO(b"c" * 100)
            output.seek(100)
            file_download.http_get(
                "https://example.invalid/weights.gguf",
                output,
                resume_size = 100,
                expected_size = 380,
                tqdm_class = tqdm_class,
            )
            assert output.tell() == 380
        else:
            from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter
            with XetDownloadProgressReporter(
                reconstruction_desc = "weights",
                log_level = logging.WARNING,
                tqdm_class = tqdm_class,
                total = 500,
            ) as reporter:
                for second in range(1, 5):
                    clock[0] += 1
                    reporter.update_progress(
                        SimpleNamespace(
                            total_transfer_bytes_completed = second * 70,
                            total_transfer_bytes_completion_rate = 70,
                            total_bytes_completed = 0,
                            total_bytes_completion_rate = 0,
                            total_bytes = 500,
                        )
                    )
        return str(tmp_path / "weights.gguf")

    monkeypatch.setattr(snapshot, "hf_hub_download", download)
    beat = tmp_path / "beat"
    snapshot.snapshot_download(
        "Org/Model",
        revision = "a" * 40,
        cache_dir = tmp_path / "cache",
        token = False,
        tqdm_class = hf_download._progress_class(str(beat)),
        max_workers = 1,
    )
    reading = download_heartbeat.read_transfer(str(beat))
    assert reading["network_bytes"] == 280
    assert reading["elapsed_seconds"] == 4


def test_transfer_age_uses_last_network_update_not_reconstruction(monkeypatch, tmp_path):
    clock = [100.0]
    monkeypatch.setattr(download_heartbeat.time, "time", lambda: clock[0])
    beat = tmp_path / "beat"
    writer = download_heartbeat.HeartbeatWriter(str(beat), interval = 0)
    writer.add(70, transfer = True)
    clock[0] += 20
    writer.add(100)
    assert download_heartbeat.read_transfer(str(beat))["age_seconds"] == 20
    assert download_heartbeat.read(str(beat)) == 170


@pytest.mark.parametrize("payload", ["broken", "{}"])
def test_unavailable_transfer_statistics(tmp_path, payload):
    beat = tmp_path / "beat"
    beat.write_text(payload)
    assert download_heartbeat.read_transfer(str(beat)) is None


def test_progress_api_attaches_only_current_live_worker_statistics(monkeypatch, tmp_path):
    beat = tmp_path / "beat"
    writer = download_heartbeat.HeartbeatWriter(str(beat), interval = 0)
    writer.add(70_000_000, transfer = True)
    proc = _Proc(beat)
    registry = SimpleNamespace(get_process = lambda key: proc)
    disk = {
        "downloaded_bytes": 24,
        "completed_bytes": 0,
        "complete_on_disk": False,
        "expected_bytes": 194,
        "progress": 24 / 194,
    }
    monkeypatch.setattr(snapshot_progress, "compute_snapshot_progress", lambda **kw: dict(disk))

    def progress():
        return asyncio.run(
            snapshot_progress.snapshot_progress_response(
                repo_type = "model",
                repo_id = "Org/Model",
                job_key = "key",
                expected_bytes = 194,
                hf_token = None,
                registry = registry,
                metadata_resolver = lambda *args: (194, frozenset()),
            )
        )

    result = progress()
    assert result["transfer_progress"]["network_bytes"] == 70_000_000
    assert result["transfer_progress"]["attempt_id"] == beat.name
    assert {key: result[key] for key in disk} == disk
    from hub.schemas.downloads import DownloadProgressResponse

    assert (
        DownloadProgressResponse(**result).model_dump()["transfer_progress"]["network_bytes"]
        == 70_000_000
    )

    proc.rc = 0
    assert progress() == disk
    proc.rc = None
    registry.get_process = lambda key: None
    assert progress() == disk
    calls = iter([proc, _Proc(beat)])
    registry.get_process = lambda key: next(calls)
    assert progress() == disk
    registry.get_process = lambda key: proc
    beat.write_text("broken")
    assert progress() == disk
