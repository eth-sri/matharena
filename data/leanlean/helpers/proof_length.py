#!/usr/bin/env python3
"""Measure the current project's Lean source tokens using its pinned tooling."""
import os

os.execvp("proof-length", ["proof-length"])
