# Single-V620 (gfx1030 / RDNA2) measurement harness for Phase 0-2.
#
# Everything in this package imports GPU modules lazily so that --help,
# comparison and the CPU-only tests run on a host with a CPU-only torch
# (or no torch at all, for the pure-stdlib pieces).
