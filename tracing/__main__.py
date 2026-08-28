"""Runnable entry point: ``python -m tracing`` -> the standalone demo/driver.

Wraps a simple in-process fake memo, drives a scripted BUILD + a couple of
RETRIEVE calls, produces a sample per-user git repo under a temp dir, and prints
the resulting git log. Zero external dependencies. See :mod:`tracing.demo`.
"""

from __future__ import annotations

from tracing.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
