"""Compute kernel, cokernel, and image persistence from maps of F2 chain complexes."""

from .phimaker import sixpack, sixpack_from_inclusion

__all__ = ["sixpack", "sixpack_from_inclusion"]
