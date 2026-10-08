import subprocess

import pytest

from scripts.pods import harness


@pytest.mark.parametrize("output", ["starting instance 123.\n", '{"success": true}', "{'success': True}"])
def test_start_accepts_cli_or_json_acknowledgment(monkeypatch, output):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, output, ""))
    harness.vast("start", "instance", 123)


@pytest.mark.parametrize("output", ["", "Insufficient available resources",
                                    '{"success": false, "msg": "resources unavailable"}'])
def test_start_refuses_exit_zero_without_success(monkeypatch, output):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, output, ""))
    with pytest.raises(RuntimeError, match="start rejected"):
        harness.vast("start", "instance", 123)


def test_rejection_does_not_echo_provider_secrets():
    assert str(harness.provider_rejection("start", "resources unavailable; token=do-not-copy")) == \
        "vastai start rejected: resources unavailable"


@pytest.mark.parametrize("action", ["create", "start", "stop", "destroy"])
def test_stderr_http_error_is_not_a_successful_mutation(monkeypatch, action):
    diagnostic = '{"error": true, "status_code": 402, "msg": "insufficient credit; token=do-not-copy"}'
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, "", diagnostic))
    with pytest.raises(RuntimeError, match=rf"^vastai {action} rejected \(HTTP 402\): billing or balance$"):
        harness.vast(action, "instance", 123)


@pytest.mark.parametrize("now", [5, 15])
def test_obsolete_guard_cannot_stop_a_rearmed_stage(monkeypatch, now):
    monkeypatch.setattr(harness.sys, "argv", ["harness", "_guard", "123", "10"])
    monkeypatch.setattr(harness.time, "time", lambda: now)
    monkeypatch.setattr(harness, "state", lambda: {"id": 123, "deadline": 20})
    monkeypatch.setattr(harness, "vast", lambda *a: pytest.fail("obsolete guard called provider"))
    harness.main()


def test_guard_carries_its_selected_rental_record(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(harness, "STATE", tmp_path / "second" / "pod.json")
    monkeypatch.setattr(harness.sys, "platform", "darwin")
    monkeypatch.setattr(harness.subprocess, "run", lambda args, **kw: commands.append(args))
    harness.arm_guard({"provider": "vast", "id": 123}, 0.5)
    command = next(args for args in commands if args[:2] == ["launchctl", "submit"])
    assert f"NINFER_POD_STATE={harness.STATE}" in command


@pytest.mark.parametrize("actual,intended,expected", [
    ("exited", "stopped", True), ("stopped", "stopped", True),
    ("exited", "running", False), ("running", "stopped", False),
    ("exited", None, False), (None, "stopped", False),
])
def test_stop_requires_both_terminal_compute_and_canceled_start(actual, intended, expected):
    assert harness.rental_stopped(actual, intended) is expected


@pytest.mark.parametrize("entry", ["stop", "_guard"])
def test_explicit_stop_and_guard_cancel_a_queued_start(monkeypatch, entry):
    pod = {"provider": "vast", "id": 123, "deadline": 10,
           "actual_status": "exited", "intended_status": "running"}
    stopped = []
    monkeypatch.setattr(harness.sys, "argv", ["harness", entry] +
                        (["123", "10"] if entry == "_guard" else []))
    monkeypatch.setattr(harness.time, "time", lambda: 15)
    monkeypatch.setattr(harness.time, "sleep", lambda _: None)
    monkeypatch.setattr(harness, "state", lambda: pod.copy())
    monkeypatch.setattr(harness, "endpoint", lambda p:
                        {**p, "status": pod["actual_status"], "intended_status": pod["intended_status"]})
    monkeypatch.setattr(harness, "ssh", lambda *a: pytest.fail("queued start needs no SSH"))

    def provider(*args):
        if args == ("show", "instances"):
            return [pod.copy()]
        assert args == ("stop", "instance", 123)
        stopped.append(args)
        pod["intended_status"] = "stopped"
        return {}

    monkeypatch.setattr(harness, "vast", provider)
    harness.main()
    assert len(stopped) == 1
