import hashlib
import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_wrapper.cli import main
from harness_wrapper.installation import (
    DEFAULT_CLI_VERSION,
    SUPPORTED_CLI_RELEASES,
    cli_bin_dir,
    cli_environment,
    ensure_cli,
    install_clis,
    installed_versions,
    resolve_cli_release,
)
from harness_wrapper.models.oauth import OAuthConfig


def test_releases_default_to_latest_and_are_complete() -> None:
    assert set(SUPPORTED_CLI_RELEASES) == {
        "claude-code",
        "codex-cli",
        "antigravity-cli",
        "kimi-code",
        "opencode",
        "qwen-code",
        "deepcode",
        "muse-code",
        "grok-build",
        "mistral-vibe",
    }
    assert all(
        release.version == DEFAULT_CLI_VERSION for release in SUPPORTED_CLI_RELEASES.values()
    )


def test_latest_resolves_through_package_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def metadata(url: str):
        urls.append(url)
        return {"version": "9.8.7"}

    monkeypatch.setattr("harness_wrapper.installation._fetch_json", metadata)

    release = resolve_cli_release("kimi-code")

    assert release.version == "9.8.7"
    assert urls == ["https://registry.npmjs.org/%40moonshot-ai%2Fkimi-code/latest"]


def test_antigravity_latest_resolves_through_github(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def metadata(url: str):
        urls.append(url)
        return {"tag_name": "v1.1.26"}

    monkeypatch.setattr("harness_wrapper.installation._fetch_json", metadata)

    release = resolve_cli_release("antigravity-cli")

    assert release.version == "1.1.26"
    assert urls == [
        "https://api.github.com/repos/google-antigravity/antigravity-cli/releases/latest"
    ]


def test_dry_run_reports_pinned_native_artifact() -> None:
    urls = install_clis(["codex-cli"], versions={"codex-cli": "0.147.0"}, dry_run=True)
    assert urls == [
        "https://github.com/openai/codex/releases/download/rust-v0.147.0/"
        "codex-package-x86_64-unknown-linux-musl.tar.gz"
    ]


def test_private_bin_is_prepended(tmp_path: Path) -> None:
    env = cli_environment(tmp_path, base={"PATH": "/usr/bin", "KEEP": "yes"})
    assert env["PATH"] == f"{cli_bin_dir(tmp_path)}:/usr/bin"
    assert env["KEEP"] == "yes"


def test_unknown_cli_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown CLI"):
        install_clis(["not-real"], dry_run=True)


def test_missing_private_install_reports_none(tmp_path: Path) -> None:
    assert installed_versions(tmp_path) == {
        "claude-code": None,
        "codex-cli": None,
        "antigravity-cli": None,
        "kimi-code": None,
        "opencode": None,
        "qwen-code": None,
        "deepcode": None,
        "muse-code": None,
        "grok-build": None,
        "mistral-vibe": None,
    }


def test_cli_dry_run_without_names_selects_all_releases(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "install-clis",
                "--dry-run",
                "--version",
                "claude-code=2.1.231",
                "--version",
                "codex-cli=0.147.0",
                "--version",
                "kimi-code=0.36.0",
                "--version",
                "antigravity-cli=1.1.26",
                "--version",
                "qwen-code=0.23.0",
                "--version",
                "opencode=1.18.27",
                "--version",
                "deepcode=0.3.1",
                "--version",
                "muse-code=1.0.3-R2198.1",
                "--version",
                "grok-build=1.0.40",
                "--version",
                "mistral-vibe=2.26.0",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "/2.1.231/linux-x64/claude" in output
    assert "/rust-v0.147.0/codex-package-x86_64-unknown-linux-musl.tar.gz" in output
    assert "/0.36.0/kimi-code-linux-x64" in output

    assert "/antigravity-cli/releases/download/1.1.26/" in output
    assert "/agy_cli_linux_x64.tar.gz" in output
    assert "npm:@qwen-code/qwen-code@0.23.0" in output
    assert "npm:opencode-ai@1.18.27" in output
    assert "npm:@vegamo/deepcode-cli@0.3.1" in output
    assert "/mistral-vibe/releases/download/v2.26.0/vibe-linux-x86_64-2.26.0.zip" in output


def test_npm_distribution_is_staged_with_its_package_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def npm_install(command, *, check):
        assert check is True
        calls.append(list(command))
        package = Path(command[command.index("--prefix") + 1])
        module = package / "lib" / "node_modules" / "fake-qwen"
        module.mkdir(parents=True)
        (module / "cli.js").write_text("#!/usr/bin/env node\n", encoding="utf-8")
        bin_dir = package / "bin"
        bin_dir.mkdir()
        (bin_dir / "qwen").symlink_to("../lib/node_modules/fake-qwen/cli.js")

    monkeypatch.setattr("harness_wrapper.installation.subprocess.run", npm_install)

    urls = install_clis(
        ["qwen-code"],
        prefix=tmp_path,
        versions={"qwen-code": "0.23.0"},
        npm="test-npm",
    )

    assert urls == ["npm:@qwen-code/qwen-code@0.23.0"]
    assert calls[0][0] == "test-npm"
    assert calls[0][-1] == "@qwen-code/qwen-code@0.23.0"
    assert (tmp_path / "bin" / "qwen").is_symlink()
    assert (tmp_path / "bin" / "qwen").resolve().read_text() == "#!/usr/bin/env node\n"


def test_install_always_downloads_and_replaces_native_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: list[str] = []
    payloads = iter((b"first", b"second"))

    def download(url: str, destination: Path) -> None:
        downloads.append(url)
        destination.write_bytes(next(payloads))

    def manifest(url: str):
        payload = b"first" if not downloads else b"second"
        return {
            "platforms": {
                "linux-x64": {
                    "filename": "kimi-code-linux-x64",
                    "checksum": hashlib.sha256(payload).hexdigest(),
                }
            }
        }

    monkeypatch.setattr("harness_wrapper.installation._download", download)
    monkeypatch.setattr("harness_wrapper.installation._fetch_json", manifest)

    versions = {"kimi-code": "0.36.0"}
    install_clis(["kimi-code"], prefix=tmp_path, versions=versions)
    executable = tmp_path / "bin" / "kimi"
    assert executable.read_bytes() == b"first"
    install_clis(["kimi-code"], prefix=tmp_path, versions=versions)
    assert executable.read_bytes() == b"second"
    assert len(downloads) == 2
    assert executable.stat().st_mode & 0o777 == 0o755


def _codex_archive(files: dict[str, tuple[bytes, int]]) -> bytes:
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for name, (content, mode) in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = mode
            archive.addfile(member, io.BytesIO(content))
    return archive_buffer.getvalue()


def _mock_codex_download(archive_bytes: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    filename = "codex-package-x86_64-unknown-linux-musl.tar.gz"
    monkeypatch.setattr(
        "harness_wrapper.installation._fetch_json",
        lambda url: {
            "assets": [
                {
                    "name": filename,
                    "digest": f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}",
                }
            ]
        },
    )
    monkeypatch.setattr(
        "harness_wrapper.installation._download",
        lambda url, destination: destination.write_bytes(archive_bytes),
    )


def test_antigravity_archive_is_verified_and_installs_agy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _codex_archive({"antigravity": (b"native-antigravity", 0o755)})
    filename = "agy_cli_linux_x64.tar.gz"
    seen_urls: list[str] = []

    monkeypatch.setattr(
        "harness_wrapper.installation._fetch_json",
        lambda url: {
            "assets": [
                {
                    "name": filename,
                    "digest": f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}",
                }
            ]
        },
    )

    def download(url: str, destination: Path) -> None:
        seen_urls.append(url)
        destination.write_bytes(archive_bytes)

    monkeypatch.setattr("harness_wrapper.installation._download", download)

    install_clis(
        ["antigravity-cli"],
        prefix=tmp_path,
        versions={"antigravity-cli": "1.1.26"},
    )

    executable = tmp_path / "bin" / "agy"
    assert executable.read_bytes() == b"native-antigravity"
    assert executable.stat().st_mode & 0o777 == 0o755
    assert seen_urls == [
        "https://github.com/google-antigravity/antigravity-cli/releases/download/"
        "1.1.26/agy_cli_linux_x64.tar.gz"
    ]


def test_codex_package_is_verified_and_installs_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _codex_archive(
        {
            "codex-package.json": (b"{}", 0o644),
            "bin/codex": (b"native-codex", 0o700),
            "bin/codex-code-mode-host": (b"code-mode-host", 0o700),
            "codex-resources/bwrap": (b"bubblewrap", 0o700),
            "codex-path/rg": (b"ripgrep", 0o700),
        }
    )
    _mock_codex_download(archive_bytes, monkeypatch)

    install_clis(["codex-cli"], prefix=tmp_path, versions={"codex-cli": "0.147.0"})
    assert (tmp_path / "bin" / "codex").read_bytes() == b"native-codex"
    assert (tmp_path / "bin" / "codex-code-mode-host").read_bytes() == b"code-mode-host"
    assert (tmp_path / "codex-resources" / "bwrap").read_bytes() == b"bubblewrap"
    assert (tmp_path / "codex-path" / "rg").read_bytes() == b"ripgrep"
    assert (tmp_path / "codex-package.json").read_bytes() == b"{}"
    assert (tmp_path / "bin" / "codex").stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "bin" / "codex-code-mode-host").stat().st_mode & 0o777 == 0o755


def test_codex_package_requires_host_before_replacing_existing_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "bin" / "codex"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"existing-codex")
    archive_bytes = _codex_archive({"bin/codex": (b"new-codex", 0o755)})
    _mock_codex_download(archive_bytes, monkeypatch)

    with pytest.raises(RuntimeError, match="bin/codex-code-mode-host"):
        install_clis(["codex-cli"], prefix=tmp_path, versions={"codex-cli": "0.147.0"})

    assert executable.read_bytes() == b"existing-codex"


def test_codex_package_rejects_unsafe_archive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _codex_archive(
        {
            "bin/codex": (b"native-codex", 0o755),
            "bin/codex-code-mode-host": (b"code-mode-host", 0o755),
            "../escaped": (b"unsafe", 0o644),
        }
    )
    _mock_codex_download(archive_bytes, monkeypatch)

    with pytest.raises(RuntimeError, match="unsafe path"):
        install_clis(["codex-cli"], prefix=tmp_path, versions={"codex-cli": "0.147.0"})

    assert not (tmp_path / "escaped").exists()


def test_ensure_cli_caches_exact_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installs: list[tuple[list[str], Path, dict[str, str]]] = []

    def install(names, *, prefix, versions, **kwargs):
        target = Path(prefix)
        executable = target / "bin" / "kimi"
        executable.parent.mkdir(parents=True)
        executable.write_text("fake")
        executable.chmod(0o755)
        installs.append((list(names), target, dict(versions)))
        return []

    monkeypatch.setattr("harness_wrapper.installation.install_clis", install)
    monkeypatch.setattr("harness_wrapper.installation._installed_version", lambda path: "0.36.0")

    first = ensure_cli("kimi-code", "0.36.0", cache_root=tmp_path)
    second = ensure_cli("kimi-code", "0.36.0", cache_root=tmp_path)

    assert first == second
    assert first.executable.is_file()
    assert first.prefix == tmp_path / "kimi-code" / "0.36.0"
    assert len(installs) == 1


def test_cli_login_lists_oauth_providers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "harness_wrapper.cli._oauth_logged_in",
        lambda provider: provider == "anthropic",
    )
    assert main(["login", "--list"]) == 0
    output = capsys.readouterr().out
    assert "❌ openai" in output
    assert "✅ anthropic" in output
    assert "❌ kimi" in output


def test_cli_login_authenticates_selected_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str]] = []
    configs: list[OAuthConfig] = []
    credential = SimpleNamespace(extra={"cli_managed": True})

    class LoginModel:
        def ensure_authenticated(self):
            calls.append(("ensure", "openai"))
            return credential

        def login(self):
            calls.append(("login", "openai"))
            return credential

    def create(model: str, *, oauth: str, oauth_config: OAuthConfig):
        calls.append((model, oauth))
        configs.append(oauth_config)
        return LoginModel()

    monkeypatch.setattr("harness_wrapper.cli.create_model", create)
    assert main(["login", "openai"]) == 0
    assert calls == [("oauth-login", "openai"), ("ensure", "openai")]
    login_command = tuple(configs[0].login_command or ())
    assert Path(login_command[0]).name == "codex"
    assert login_command[1:] == ("login", "--device-auth")
    assert "Authenticated with openai" in capsys.readouterr().out


def test_cli_openai_can_request_local_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    credential = SimpleNamespace(extra={})
    configs: list[OAuthConfig] = []

    def create(*args, oauth_config: OAuthConfig, **kwargs):
        configs.append(oauth_config)
        return SimpleNamespace(ensure_authenticated=lambda: credential)

    monkeypatch.setattr("harness_wrapper.cli.create_model", create)
    assert main(["login", "openai", "--local-callback"]) == 0
    login_command = tuple(configs[0].login_command or ())
    assert Path(login_command[0]).name == "codex"
    assert login_command[1:] == ("login",)


def test_cli_login_force_runs_provider_login(monkeypatch: pytest.MonkeyPatch) -> None:
    credential = SimpleNamespace(extra={})
    login_calls: list[bool] = []
    model = SimpleNamespace(
        ensure_authenticated=lambda: credential,
        login=lambda: login_calls.append(True) or credential,
    )
    configs: list[OAuthConfig] = []

    def create(*args, oauth_config: OAuthConfig, **kwargs):
        configs.append(oauth_config)
        return model

    monkeypatch.setattr("harness_wrapper.cli.create_model", create)
    assert main(["login", "anthropic", "--force"]) == 0
    assert login_calls == [True]
    assert configs[0].auto_login is False


def test_cli_login_rejects_unknown_provider(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["login", "unknown"]) == 2
    assert "Unknown OAuth provider" in capsys.readouterr().err


def test_cli_rejects_local_callback_for_non_openai(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["login", "kimi", "--local-callback"]) == 2
    assert "only supported for the openai" in capsys.readouterr().err


@pytest.mark.parametrize(("target", "expected"), [
    ("linux-x64", "linux-x86_64"), ("linux-arm64", "linux-aarch64"),
    ("darwin-x64", "macos-x86_64"), ("darwin-arm64", "macos-aarch64"),
])
def test_grok_platform_names(target, expected):
    from harness_wrapper.installation import _grok_target
    assert _grok_target(target) == expected


def test_grok_latest_resolves_from_official_channel(monkeypatch):
    def fetch(request, timeout):
        assert request.full_url == "https://x.ai/cli/stable"
        return io.BytesIO(b"1.0.40\n")
    monkeypatch.setattr("harness_wrapper.installation.urllib.request.urlopen", fetch)
    release = resolve_cli_release("grok-build")
    assert release.version == "1.0.40" and release.executable == "grok"


def test_grok_install_checks_version_before_committing(monkeypatch, tmp_path):
    def download(url, path):
        assert url == "https://x.ai/cli/grok-1.0.40-linux-x86_64"
        path.write_text("test executable")
    monkeypatch.setattr("harness_wrapper.installation._download", download)
    monkeypatch.setattr("harness_wrapper.installation._installed_version", lambda p: "1.0.39")
    with pytest.raises(RuntimeError, match="does not report 1.0.40"):
        install_clis(["grok-build"], prefix=tmp_path, versions={"grok-build": "1.0.40"})
    assert not (tmp_path / "bin/grok").exists()
    monkeypatch.setattr("harness_wrapper.installation._installed_version", lambda p: "1.0.40")
    install_clis(["grok-build"], prefix=tmp_path, versions={"grok-build": "1.0.40"})
    assert (tmp_path / "bin/grok").read_text() == "test executable"


def _vibe_archive(entries: dict[str, tuple[bytes, int]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for name, (content, mode) in entries.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o100000 | mode) << 16
            bundle.writestr(info, content)
    return output.getvalue()


def test_vibe_latest_resolves_through_github(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def metadata(url: str):
        urls.append(url)
        return {"tag_name": "v2.26.0"}

    monkeypatch.setattr("harness_wrapper.installation._fetch_json", metadata)
    release = resolve_cli_release("mistral-vibe")
    assert release.version == "2.26.0" and release.executable == "vibe"
    assert urls == ["https://api.github.com/repos/mistralai/mistral-vibe/releases/latest"]


def test_vibe_bundle_is_verified_and_installs_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = _vibe_archive(
        {
            "vibe": (b"native-vibe", 0o755),
            "_internal/libpython3.12.so.1.0": (b"python", 0o755),
            "_internal/vibe/__init__.py": (b"", 0o644),
        }
    )
    filename = "vibe-linux-x86_64-2.26.0.zip"
    seen_urls: list[str] = []
    monkeypatch.setattr("harness_wrapper.installation._platforms", lambda: ("linux-x64",) * 3)
    monkeypatch.setattr(
        "harness_wrapper.installation._fetch_json",
        lambda url: {
            "assets": [
                {"name": filename, "digest": f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}"}
            ]
        },
    )

    def download(url: str, destination: Path) -> None:
        seen_urls.append(url)
        destination.write_bytes(archive_bytes)

    monkeypatch.setattr("harness_wrapper.installation._download", download)
    (tmp_path / "bin/_internal").mkdir(parents=True)
    (tmp_path / "bin/_internal/stale.py").write_text("old")
    install_clis(["mistral-vibe"], prefix=tmp_path, versions={"mistral-vibe": "2.26.0"})

    executable = tmp_path / "bin" / "vibe"
    assert executable.read_bytes() == b"native-vibe"
    assert executable.stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "bin/_internal/vibe/__init__.py").stat().st_mode & 0o777 == 0o644
    assert not (tmp_path / "bin/_internal/stale.py").exists()
    assert seen_urls == [
        "https://github.com/mistralai/mistral-vibe/releases/download/v2.26.0/" + filename
    ]


@pytest.mark.parametrize("name", ["../vibe", "/vibe", "other/tool"])
def test_vibe_bundle_rejects_unexpected_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from harness_wrapper.installation import _extract_vibe_package

    archive = tmp_path / "vibe.zip"
    archive.write_bytes(_vibe_archive({"vibe": (b"x", 0o755), name: (b"x", 0o755)}))
    with pytest.raises(RuntimeError):
        _extract_vibe_package(archive, tmp_path / "out", "https://example.invalid/vibe.zip")
