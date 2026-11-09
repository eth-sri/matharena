"""Recognize compiled artifacts without relying on arbitrary directory names."""
BUILD_ARTIFACT_SUFFIXES = (
    ".olean", ".olean.private", ".olean.server", ".ilean", ".ir",
    ".olean.hash", ".olean.private.hash", ".olean.server.hash", ".ilean.hash",
    ".ir.hash", ".ir.sig", ".ir.sig.hash", ".c.o.export", ".c.o.export.hash",
    ".o", ".obj", ".a", ".so", ".dylib", ".dll", ".pyc", ".pyo",
)


def is_build_artifact(name: str) -> bool:
    return name.rsplit("/", 1)[-1].endswith(BUILD_ARTIFACT_SUFFIXES)
