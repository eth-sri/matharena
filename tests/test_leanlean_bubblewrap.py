from leanlean.bubblewrap import command


def test_comparator_clean_environment_gets_thread_default(monkeypatch):
    monkeypatch.delenv("LEAN_NUM_THREADS", raising=False)
    args = ["--ro-bind", "/", "/", "--clearenv", "--", "lake", "build"]
    assert command(args) == [
        "/usr/bin/bwrap", "--ro-bind", "/", "/", "--clearenv",
        "--setenv", "LEAN_NUM_THREADS", "8", "--", "lake", "build",
    ]


def test_explicit_thread_setting_and_child_arguments_are_preserved(monkeypatch):
    monkeypatch.setenv("LEAN_NUM_THREADS", "3")
    args = ["--clearenv", "--setenv", "LEAN_NUM_THREADS", "2", "--", "echo", "--clearenv"]
    assert command(args) == ["/usr/bin/bwrap", *args]
    assert command(["--clearenv", "--", "lean"])[-5:] == ["--setenv", "LEAN_NUM_THREADS", "3", "--", "lean"]
    assert command(["--version"]) == ["/usr/bin/bwrap", "--version"]
