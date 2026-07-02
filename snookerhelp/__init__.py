"""SnookerHelp v1 package.

The v1 package is introduced beside the existing prototype `vision` package.
Prototype modules remain available during migration; new code should target the
schemas and package boundaries here.
"""

from .core.schema import TableState

__all__ = ["TableState"]
