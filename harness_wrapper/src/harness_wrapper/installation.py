"""Download supported native agent CLIs into a stable user-owned directory."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

DEFAULT_CLI_HOME = Path.home() / ".harness-wrapper" / "clis" / "native"
DEFAULT_CLI_CACHE = Path.home() / ".cache" / "harness-wrapper" / "clis"
DEFAULT_CLI_VERSION = "latest"


@dataclass(frozen=True)
class CLIRelease:
    """A native CLI package and either a requested or resolved version."""

    name: str
    package: str
    version: str
    executable: str

    @property
    def spec(self) -> str:
        """Retain the corresponding npm spec as useful release metadata."""

        return f"{self.package}@{self.version}"


# Adapters define package identity, not release policy. Callers may pin an exact
# version; an omitted version resolves the package registry's latest tag.
SUPPORTED_CLI_RELEASES: Mapping[str, CLIRelease] = {
    "claude-code": CLIRelease(
        "claude-code", "@anthropic-ai/claude-code", DEFAULT_CLI_VERSION, "claude"
    ),
    "codex-cli": CLIRelease("codex-cli", "@openai/codex", DEFAULT_CLI_VERSION, "codex"),
    "kimi-code": CLIRelease("kimi-code", "@moonshot-ai/kimi-code", DEFAULT_CLI_VERSION, "kimi"),
    "antigravity-cli": CLIRelease(
        "antigravity-cli",
        "google-antigravity/antigravity-cli",
        DEFAULT_CLI_VERSION,
        "agy",
    ),
    "grok-build": CLIRelease("grok-build", "xai/grok-build", DEFAULT_CLI_VERSION, "grok"),
    "muse-code": CLIRelease("muse-code", "meta/muse-code", DEFAULT_CLI_VERSION, "muse"),
    "qwen-code": CLIRelease("qwen-code", "@qwen-code/qwen-code", DEFAULT_CLI_VERSION, "qwen"),
    "opencode": CLIRelease("opencode", "opencode-ai", DEFAULT_CLI_VERSION, "opencode"),
    "deepcode": CLIRelease("deepcode", "@vegamo/deepcode-cli", DEFAULT_CLI_VERSION, "deepcode"),
    "mistral-vibe": CLIRelease(
        "mistral-vibe", "mistralai/mistral-vibe", DEFAULT_CLI_VERSION, "vibe"
    ),
}

_CLAUDE_DOWNLOAD_BASE = "https://downloads.claude.ai/claude-code-releases"
_CODEX_RELEASE_BASE = "https://github.com/openai/codex/releases/download"
_CODEX_RELEASE_API = "https://api.github.com/repos/openai/codex/releases/tags"
_KIMI_DOWNLOAD_BASE = "https://code.kimi.com/kimi-code/binaries"
_ANTIGRAVITY_RELEASE_BASE = (
    "https://github.com/google-antigravity/antigravity-cli/releases/download"
)
_ANTIGRAVITY_RELEASE_API = (
    "https://api.github.com/repos/google-antigravity/antigravity-cli/releases"
)
_GROK_DOWNLOAD_BASE = "https://x.ai/cli"
_MUSE_CHANNEL_URL = "https://api.meta.ai/muse-code/channels/muse-stable"
_MUSE_DOWNLOAD_BASE = "https://lookaside.facebook.com/lookaside/muse/download/?channel=muse"
_VIBE_RELEASE_BASE = "https://github.com/mistralai/mistral-vibe/releases/download"
_VIBE_RELEASE_API = "https://api.github.com/repos/mistralai/mistral-vibe/releases"
_NPM_REGISTRY = "https://registry.npmjs.org"
_VERSION_RE = re.compile(r"^\d+(?:\.\d+){2}(?:[-+][0-9A-Za-z.-]+)?$")
_NPM_DISTRIBUTIONS = frozenset({"qwen-code", "opencode", "deepcode"})


@dataclass(frozen=True)
class InstalledCLI:
    """A resolved native CLI in a version-specific host cache."""

    release: CLIRelease
    requested_version: str
    prefix: Path

    @property
    def executable(self) -> Path:
        return cli_bin_dir(self.prefix) / self.release.executable


def cli_bin_dir(prefix: Path | str | None = None) -> Path:
    """Return the stable directory containing native CLI executables."""

    root = Path(prefix).expanduser() if prefix is not None else DEFAULT_CLI_HOME
    return root / "bin"


def cli_environment(
    prefix: Path | str | None = None,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that resolves the privately installed CLIs first."""

    env = dict(os.environ if base is None else base)
    current = env.get("PATH", "")
    private_bin = str(cli_bin_dir(prefix))
    env["PATH"] = private_bin if not current else os.pathsep.join((private_bin, current))
    return env


def _platforms() -> tuple[str, str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arches = {
        "x86_64": ("x64", "x86_64"),
        "amd64": ("x64", "x86_64"),
        "aarch64": ("arm64", "aarch64"),
        "arm64": ("arm64", "aarch64"),
    }
    try:
        short_arch, rust_arch = arches[machine]
    except KeyError as error:
        raise RuntimeError(f"unsupported CLI architecture: {machine}") from error
    if system == "darwin":
        return f"darwin-{short_arch}", f"{rust_arch}-apple-darwin", f"darwin-{short_arch}"
    if system == "linux":
        return f"linux-{short_arch}", f"{rust_arch}-unknown-linux-musl", f"linux-{short_arch}"
    raise RuntimeError(f"native CLI installation is not supported on {platform.system()}")


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "harness-wrapper"})
    with urllib.request.urlopen(request, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid JSON document downloaded from {url}")
    return value


def _validate_version(version: str) -> str:
    normalized = version.strip()
    if normalized != DEFAULT_CLI_VERSION and not _VERSION_RE.fullmatch(normalized):
        raise ValueError(
            f"CLI version must be 'latest' or an exact semantic version, got {version!r}"
        )
    return normalized


def resolve_cli_release(name: str, version: str = DEFAULT_CLI_VERSION) -> CLIRelease:
    """Resolve a supported CLI and turn latest into an exact version."""

    try:
        template = SUPPORTED_CLI_RELEASES[name]
    except KeyError as error:
        raise ValueError(f"unknown CLI: {name}") from error
    requested = _validate_version(version)
    if requested == DEFAULT_CLI_VERSION and name == "antigravity-cli":
        metadata = _fetch_json(f"{_ANTIGRAVITY_RELEASE_API}/latest")
        tag = metadata.get("tag_name")
        if not isinstance(tag, str):
            raise RuntimeError("latest version is absent for Antigravity CLI")
        requested = _validate_version(tag.removeprefix("v"))
    elif requested == DEFAULT_CLI_VERSION and name == "grok-build":
        request = urllib.request.Request(
            f"{_GROK_DOWNLOAD_BASE}/stable", headers={"User-Agent": "harness-wrapper"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            requested = _validate_version(response.read().decode().strip())
    elif requested == DEFAULT_CLI_VERSION and name == "muse-code":
        requested = _validate_version(str(_fetch_json(_MUSE_CHANNEL_URL)["version"]))
    elif requested == DEFAULT_CLI_VERSION and name == "mistral-vibe":
        tag = _fetch_json(f"{_VIBE_RELEASE_API}/latest").get("tag_name")
        if not isinstance(tag, str):
            raise RuntimeError("latest version is absent for Mistral Vibe")
        requested = _validate_version(tag.removeprefix("v"))
    elif requested == DEFAULT_CLI_VERSION:
        package = quote(template.package, safe="")
        metadata = _fetch_json(f"{_NPM_REGISTRY}/{package}/latest")
        resolved = metadata.get("version")
        if not isinstance(resolved, str):
            raise RuntimeError(f"latest version is absent for {template.package}")
        requested = _validate_version(resolved)
    return CLIRelease(template.name, template.package, requested, template.executable)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "harness-wrapper"})
    with (
        urllib.request.urlopen(request, timeout=300) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_download(url: str, checksum: str, destination: Path) -> None:
    _download(url, destination)
    actual = _checksum(destination)
    if actual != checksum:
        raise RuntimeError(f"checksum mismatch for {url}: expected {checksum}, downloaded {actual}")


def _manifest_platform(manifest: Mapping[str, Any], target: str, url: str) -> Mapping[str, Any]:
    platforms = manifest.get("platforms")
    entry = platforms.get(target) if isinstance(platforms, dict) else None
    if not isinstance(entry, dict):
        raise RuntimeError(f"platform {target!r} is absent from {url}")
    return entry


def _grok_target(native_target: str) -> str:
    system, architecture = native_target.split("-", 1)
    system = "macos" if system == "darwin" else system
    architecture = "aarch64" if architecture == "arm64" else "x86_64"
    return f"{system}-{architecture}"


def _install_grok(release: CLIRelease, target: str, temporary: Path) -> Path:
    # The official installer publishes versioned HTTPS binaries, without a checksum manifest.
    downloaded = temporary / release.executable
    _download(f"{_GROK_DOWNLOAD_BASE}/grok-{release.version}-{target}", downloaded)
    downloaded.chmod(0o755)
    if _installed_version(downloaded) != release.version:
        raise RuntimeError(f"downloaded Grok Build does not report {release.version}")
    return downloaded


def _muse_target(native_target: str) -> str:
    system, architecture = native_target.split("-", 1)
    architecture = "aarch64" if architecture == "arm64" else "x86"
    return f"{architecture}_{'macos' if system == 'darwin' else system}"


def _muse_url(version: str, filename: str) -> str:
    return f"{_MUSE_DOWNLOAD_BASE}&version={quote(version, safe='')}&file={filename}"


def _install_muse(release: CLIRelease, target: str, temporary: Path) -> Path:
    manifest = _fetch_json(_muse_url(release.version, "manifest.json"))
    if manifest.get("version") != release.version or manifest.get("checksum_algorithm") != "sha256":
        raise RuntimeError("invalid Muse release manifest")
    entry = manifest.get("artifacts", {}).get(target, {})
    url, checksum = entry.get("url"), entry.get("checksum")
    if not isinstance(url, str) or not isinstance(checksum, str):
        raise RuntimeError(f"Muse artifact/checksum missing for {target}")
    downloaded = temporary / release.executable
    _verified_download(url, checksum, downloaded)
    return downloaded


def _install_claude(release: CLIRelease, target: str, temporary: Path) -> Path:
    manifest_url = f"{_CLAUDE_DOWNLOAD_BASE}/{release.version}/manifest.json"
    entry = _manifest_platform(_fetch_json(manifest_url), target, manifest_url)
    checksum = entry.get("checksum")
    if not isinstance(checksum, str):
        raise RuntimeError(f"invalid checksum in {manifest_url}")
    downloaded = temporary / release.executable
    _verified_download(
        f"{_CLAUDE_DOWNLOAD_BASE}/{release.version}/{target}/claude",
        checksum,
        downloaded,
    )
    return downloaded


def _install_kimi(release: CLIRelease, target: str, temporary: Path) -> Path:
    manifest_url = f"{_KIMI_DOWNLOAD_BASE}/{release.version}/manifest.json"
    entry = _manifest_platform(_fetch_json(manifest_url), target, manifest_url)
    filename = entry.get("filename")
    checksum = entry.get("checksum")
    if not isinstance(filename, str) or not isinstance(checksum, str):
        raise RuntimeError(f"invalid platform entry in {manifest_url}")
    downloaded = temporary / release.executable
    _verified_download(f"{_KIMI_DOWNLOAD_BASE}/{release.version}/{filename}", checksum, downloaded)
    return downloaded


def _antigravity_target(native_target: str) -> str:
    system, architecture = native_target.split("-", 1)
    return f"{'mac' if system == 'darwin' else system}_{architecture}"


def _antigravity_filename(target: str) -> str:
    return f"agy_cli_{target}.tar.gz"


def _install_antigravity(release: CLIRelease, target: str, temporary: Path) -> Path:
    filename = _antigravity_filename(target)
    metadata_url = f"{_ANTIGRAVITY_RELEASE_API}/tags/{release.version}"
    metadata = _fetch_json(metadata_url)
    assets = metadata.get("assets")
    asset = (
        next(
            (item for item in assets if isinstance(item, dict) and item.get("name") == filename),
            None,
        )
        if isinstance(assets, list)
        else None
    )
    digest = asset.get("digest") if isinstance(asset, dict) else None
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError(f"SHA-256 digest for {filename} is absent from {metadata_url}")

    url = f"{_ANTIGRAVITY_RELEASE_BASE}/{release.version}/{filename}"
    archive = temporary / filename
    _verified_download(url, digest.removeprefix("sha256:"), archive)
    downloaded = temporary / release.executable
    found = False
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            parts = tuple(part for part in path.parts if part != ".")
            if path.is_absolute() or ".." in parts:
                raise RuntimeError(f"unsafe path {member.name!r} in {url}")
            if parts != ("antigravity",):
                continue
            if found or not member.isfile():
                raise RuntimeError(f"invalid Antigravity executable entry in {url}")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read {member.name!r} from {url}")
            with source, downloaded.open("xb") as output:
                shutil.copyfileobj(source, output)
            found = True
    if not found:
        raise RuntimeError(f"Antigravity executable is absent from {url}")
    return downloaded


def _vibe_target(native_target: str) -> str:
    system, architecture = native_target.split("-", 1)
    return f"{system}-{'aarch64' if architecture == 'arm64' else 'x86_64'}"


def _vibe_filename(target: str, version: str) -> str:
    return f"vibe-{target}-{version}.zip"


def _extract_vibe_package(archive: Path, destination: Path, url: str) -> None:
    """Extract the PyInstaller bundle: ``vibe`` plus its adjacent ``_internal`` tree."""

    bin_dir = destination / "bin"
    bin_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            parts = tuple(part for part in path.parts if part != ".")
            if path.is_absolute() or ".." in parts:
                raise RuntimeError(f"unsafe path {member.filename!r} in {url}")
            if not parts or member.is_dir():
                continue
            if parts != ("vibe",) and parts[0] != "_internal":
                raise RuntimeError(f"unexpected archive entry {member.filename!r} in {url}")
            mode = member.external_attr >> 16
            if mode and not stat.S_ISREG(mode):
                raise RuntimeError(f"unsupported archive entry {member.filename!r} in {url}")
            output = bin_dir.joinpath(*parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, output.open("xb") as target_file:
                shutil.copyfileobj(source, target_file)
            output.chmod((mode & 0o777) or 0o644)
    if not (bin_dir / "vibe").is_file():
        raise RuntimeError(f"Mistral Vibe executable is absent from {url}")


def _install_vibe(release: CLIRelease, target: str, temporary: Path) -> Path:
    tag = f"v{release.version}"
    filename = _vibe_filename(target, release.version)
    metadata_url = f"{_VIBE_RELEASE_API}/tags/{tag}"
    metadata = _fetch_json(metadata_url)
    assets = metadata.get("assets")
    asset = (
        next(
            (item for item in assets if isinstance(item, dict) and item.get("name") == filename),
            None,
        )
        if isinstance(assets, list)
        else None
    )
    digest = asset.get("digest") if isinstance(asset, dict) else None
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError(f"SHA-256 digest for {filename} is absent from {metadata_url}")

    url = f"{_VIBE_RELEASE_BASE}/{tag}/{filename}"
    archive = temporary / filename
    _verified_download(url, digest.removeprefix("sha256:"), archive)
    package = temporary / "vibe-package"
    _extract_vibe_package(archive, package, url)
    return package


def _commit_vibe_package(package: Path, root: Path) -> None:
    """Replace the bundle's runtime tree, exposing the new entrypoint last."""

    bin_dir = cli_bin_dir(root)
    internal = bin_dir / "_internal"
    if internal.exists():
        shutil.rmtree(internal)
    os.replace(package / "bin" / "_internal", internal)
    os.replace(package / "bin" / "vibe", bin_dir / "vibe")
    (bin_dir / "vibe").chmod(0o755)


def _codex_package_filename(target: str) -> str:
    return f"codex-package-{target}.tar.gz"


def _codex_package_path(member: tarfile.TarInfo, url: str) -> Path | None:
    """Return a safe, supported path from a canonical Codex package."""

    path = PurePosixPath(member.name)
    parts = tuple(part for part in path.parts if part != ".")
    if path.is_absolute() or ".." in parts:
        raise RuntimeError(f"unsafe path {member.name!r} in {url}")
    if not parts:
        return None

    allowed = (
        parts == ("codex-package.json",)
        or parts[:1] in {("codex-resources",), ("codex-path",)}
        or parts in {("bin", "codex"), ("bin", "codex-code-mode-host")}
    )
    return Path(*parts) if allowed else None


def _extract_codex_package(archive: Path, destination: Path, url: str) -> None:
    """Extract regular files from a Codex package without trusting tar paths."""

    destination.mkdir()
    extracted: set[Path] = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            relative = _codex_package_path(member, url)
            if relative is None or member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported archive entry {member.name!r} in {url}")
            if relative in extracted:
                raise RuntimeError(f"duplicate archive entry {member.name!r} in {url}")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read {member.name!r} from {url}")
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with source, output.open("xb") as target_file:
                shutil.copyfileobj(source, target_file)
            output.chmod(member.mode & 0o777)
            extracted.add(relative)

    required = (Path("bin/codex"), Path("bin/codex-code-mode-host"))
    missing = [str(path) for path in required if path not in extracted]
    if missing:
        raise RuntimeError(f"required file(s) {', '.join(missing)} are absent from {url}")


def _install_codex(release: CLIRelease, target: str, temporary: Path) -> Path:
    tag = f"rust-v{release.version}"
    filename = _codex_package_filename(target)
    metadata_url = f"{_CODEX_RELEASE_API}/{tag}"
    metadata = _fetch_json(metadata_url)
    assets = metadata.get("assets")
    asset = (
        next(
            (item for item in assets if isinstance(item, dict) and item.get("name") == filename),
            None,
        )
        if isinstance(assets, list)
        else None
    )
    digest = asset.get("digest") if isinstance(asset, dict) else None
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError(f"SHA-256 digest for {filename} is absent from {metadata_url}")

    archive = temporary / filename
    url = f"{_CODEX_RELEASE_BASE}/{tag}/{filename}"
    _verified_download(url, digest.removeprefix("sha256:"), archive)
    package = temporary / "codex-package"
    _extract_codex_package(archive, package, url)
    return package


def _commit_codex_package(package: Path, root: Path) -> None:
    """Install a validated package, exposing the new entrypoint last."""

    files = [path for path in package.rglob("*") if path.is_file()]
    entrypoint = package / "bin" / "codex"
    files.sort(key=lambda path: path == entrypoint)
    for source in files:
        relative = source.relative_to(package)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    (root / "bin" / "codex").chmod(0o755)
    (root / "bin" / "codex-code-mode-host").chmod(0o755)


def _install_npm_release(
    release: CLIRelease,
    root: Path,
    temporary: Path,
    npm: str,
) -> None:
    """Install one exact npm distribution, then merge it into the cache.

    Qwen Code is a JavaScript distribution and needs its package tree beside
    the generated ``bin`` symlink. OpenCode uses the same
    packaging channel for its platform-specific native executable. Installing
    into a temporary prefix ensures a failed npm/postinstall step cannot leave
    an apparently valid entrypoint in the shared cache.
    """

    package = temporary / release.name
    subprocess.run(
        [
            npm,
            "install",
            "--global",
            "--prefix",
            str(package),
            "--omit=dev",
            "--no-audit",
            "--no-fund",
            release.spec,
        ],
        check=True,
    )
    entrypoint = package / "bin" / release.executable
    if not entrypoint.exists():
        raise RuntimeError(f"npm package {release.spec} did not install bin/{release.executable}")

    paths = sorted(
        package.rglob("*"),
        key=lambda path: (path == entrypoint, len(path.parts), str(path)),
    )
    for source in paths:
        relative = source.relative_to(package)
        destination = root / relative
        if source.is_dir() and not source.is_symlink():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_name(f".{destination.name}.new")
        staged.unlink(missing_ok=True)
        if source.is_symlink():
            staged.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, staged, follow_symlinks=False)
        os.replace(staged, destination)


def _download_urls(releases: Sequence[CLIRelease]) -> list[str]:
    claude_target, codex_target, kimi_target = _platforms()
    antigravity_target = _antigravity_target(claude_target)
    urls: list[str] = []
    for release in releases:
        if release.name in _NPM_DISTRIBUTIONS:
            urls.append(f"npm:{release.spec}")
        elif release.name == "grok-build":
            urls.append(f"{_GROK_DOWNLOAD_BASE}/grok-{release.version}-{_grok_target(claude_target)}")
        elif release.name == "muse-code":
            target = _muse_target(claude_target).replace("_", "-")
            urls.append(_muse_url(release.version, f"muse-{target}"))
        elif release.name == "mistral-vibe":
            urls.append(
                f"{_VIBE_RELEASE_BASE}/v{release.version}/"
                f"{_vibe_filename(_vibe_target(claude_target), release.version)}"
            )
        elif release.name == "claude-code":
            urls.append(f"{_CLAUDE_DOWNLOAD_BASE}/{release.version}/{claude_target}/claude")
        elif release.name == "codex-cli":
            urls.append(
                f"{_CODEX_RELEASE_BASE}/rust-v{release.version}/"
                f"{_codex_package_filename(codex_target)}"
            )
        elif release.name == "antigravity-cli":
            urls.append(
                f"{_ANTIGRAVITY_RELEASE_BASE}/{release.version}/"
                f"{_antigravity_filename(antigravity_target)}"
            )
        else:
            urls.append(f"{_KIMI_DOWNLOAD_BASE}/{release.version}/kimi-code-{kimi_target}")
    return urls


def install_clis(
    names: Sequence[str] | None = None,
    *,
    prefix: Path | str | None = None,
    versions: Mapping[str, str] | None = None,
    npm: str = "npm",
    dry_run: bool = False,
) -> list[str]:
    """Resolve, download, and install supported CLI releases.

    Native executables and npm-generated entrypoints are replaced last in
    ``<prefix>/bin``. Versions default to latest and exact versions can be
    supplied through ``versions``. The returned list contains native artifact
    URLs or an ``npm:package@version`` descriptor for npm distributions.
    """

    selected = list(names or SUPPORTED_CLI_RELEASES)
    unknown = sorted(set(selected) - set(SUPPORTED_CLI_RELEASES))
    if unknown:
        raise ValueError(f"unknown CLI(s): {', '.join(unknown)}")
    version_overrides = dict(versions or {})
    unknown_versions = sorted(set(version_overrides) - set(selected))
    if unknown_versions:
        raise ValueError("version supplied for unselected CLI(s): " + ", ".join(unknown_versions))
    if not selected:
        return []
    releases = [
        resolve_cli_release(name, version_overrides.get(name, DEFAULT_CLI_VERSION))
        for name in selected
    ]
    urls = _download_urls(releases)
    if dry_run:
        return urls

    root = Path(prefix).expanduser() if prefix is not None else DEFAULT_CLI_HOME
    bin_dir = cli_bin_dir(root)
    bin_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    claude_target, codex_target, kimi_target = _platforms()
    antigravity_target = _antigravity_target(claude_target)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=bin_dir) as directory:
        temporary = Path(directory)
        for release in releases:
            if release.name in _NPM_DISTRIBUTIONS:
                _install_npm_release(release, root, temporary, npm)
                continue
            if release.name == "grok-build":
                downloaded = _install_grok(release, _grok_target(claude_target), temporary)
            elif release.name == "muse-code":
                downloaded = _install_muse(release, _muse_target(claude_target), temporary)
            elif release.name == "claude-code":
                downloaded = _install_claude(release, claude_target, temporary)
            elif release.name == "codex-cli":
                package = _install_codex(release, codex_target, temporary)
                _commit_codex_package(package, root)
                continue
            elif release.name == "mistral-vibe":
                package = _install_vibe(release, _vibe_target(claude_target), temporary)
                _commit_vibe_package(package, root)
                continue
            elif release.name == "antigravity-cli":
                downloaded = _install_antigravity(release, antigravity_target, temporary)
            else:
                downloaded = _install_kimi(release, kimi_target, temporary)
            downloaded.chmod(0o755)
            os.replace(downloaded, bin_dir / release.executable)
    return urls


def _installed_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(
        r"(?<![\d.])(\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?)",
        output,
    )
    if match is None:
        raise RuntimeError(f"could not parse CLI version from {output!r}")
    # Muse prints the display version first, followed by the exact release build.
    muse_build = re.search(r"\((\d+\.\d+\.\d+-R\d+(?:\.\d+)?)\)", output)
    return muse_build.group(1) if muse_build else match.group(1)


def ensure_cli(
    name: str,
    version: str = DEFAULT_CLI_VERSION,
    *,
    cache_root: Path | str | None = None,
) -> InstalledCLI:
    """Resolve and cache one exact CLI release for read-only sandbox mounting."""

    requested = _validate_version(version)
    release = resolve_cli_release(name, requested)
    root = Path(cache_root).expanduser() if cache_root is not None else DEFAULT_CLI_CACHE
    prefix = root / release.name / release.version
    executable = cli_bin_dir(prefix) / release.executable
    if executable.is_file():
        actual = _installed_version(executable)
        if actual != release.version:
            raise RuntimeError(
                f"cached {release.name} executable reports {actual}, "
                f"expected {release.version}: {executable}"
            )
        return InstalledCLI(release, requested, prefix)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{release.version}-", dir=prefix.parent))
    try:
        install_clis(
            [release.name],
            prefix=temporary,
            versions={release.name: release.version},
        )
        actual = _installed_version(cli_bin_dir(temporary) / release.executable)
        if actual != release.version:
            raise RuntimeError(
                f"downloaded {release.name} reports {actual}, expected {release.version}"
            )
        try:
            temporary.rename(prefix)
        except OSError:
            if not executable.is_file():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    actual = _installed_version(executable)
    if actual != release.version:
        raise RuntimeError(
            f"cached {release.name} executable reports {actual}, "
            f"expected {release.version}: {executable}"
        )
    return InstalledCLI(release, requested, prefix)


def installed_versions(
    prefix: Path | str | None = None,
) -> dict[str, str | None]:
    """Report versions without invoking a shell or reading user configuration."""

    bin_dir = cli_bin_dir(prefix)
    result: dict[str, str | None] = {}
    for name, release in SUPPORTED_CLI_RELEASES.items():
        executable = bin_dir / release.executable
        if not executable.exists():
            result[name] = None
            continue
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        result[name] = (completed.stdout or completed.stderr).strip() or None
    return result
