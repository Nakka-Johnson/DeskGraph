"""Shared pytest configuration."""

from __future__ import annotations

from hypothesis import settings

# Pricing calls are fast but timing on shared CI runners is not; a per-example
# deadline would make property tests flaky without catching real bugs.
settings.register_profile("deskgraph", deadline=None, print_blob=True)
settings.load_profile("deskgraph")
