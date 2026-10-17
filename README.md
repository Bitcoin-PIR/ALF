# ALF-n-t (Rust port) — ARCHIVED

**This repository is archived and no longer maintained.**

It was an independent Rust port of the **ALF-n-t** format-preserving
encryption cipher (ARM64 NEON+AES, x86_64 AES-NI, and a wasm32 backend),
written for use as a small-domain PRP backend in BitcoinPIR's HarmonyPIR
client. BitcoinPIR retired the ALF backend in May 2026 in favour of FastPRP,
and this port has not been developed since.

## Use the authors' implementation instead

The ALF cipher family is designed and maintained by its authors
(Dachao Wang, Alexander Maximov, Thomas Johansson). Please use their
reference implementation, tools and test vectors:

- **Code and test vectors:** https://github.com/0NG/ALF-tools
- **Paper:** *Introducing the ALF family: AES-NI-based length- and
  format-preserving encryption*, https://eprint.iacr.org/2025/2148

The upstream repository receives bug fixes and updated test vectors
(for example the fixes published on 2026-10-02); this port does not.

## Notes for anyone still pinning this repository

- Only the ALF-n-t variant was ever ported here. The radix-stream ALF
  variant (`alf.h` upstream) was never part of this repository.
- The code stays available read-only at its existing commits, so a
  `git` dependency pinned to a specific revision continues to resolve.
- Usage notes for the port are kept in [USAGE.md](USAGE.md) for reference.
