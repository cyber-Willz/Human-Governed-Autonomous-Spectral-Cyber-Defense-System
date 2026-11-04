#!/usr/bin/env bash
# Builds xdp/nsm-ebpf into a freestanding BPF ELF object.
#
# One-time setup (Linux host, root not required for the build itself):
#   rustup toolchain install nightly --component rust-src
#   cargo install bpf-linker      # needs a system LLVM; see bpf-linker's README
#     - Debian/Ubuntu: apt install llvm-dev libclang-dev clang
#     - Fedora:        dnf install llvm-devel clang-devel
#
# Usage:
#   ./scripts/build-ebpf.sh            # release build (default)
#   ./scripts/build-ebpf.sh --debug    # unoptimized, for `bpftool prog dump`
set -euo pipefail

PROFILE="release"
CARGO_FLAG="--release"
if [[ "${1:-}" == "--debug" ]]; then
  PROFILE="debug"
  CARGO_FLAG=""
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../xdp/nsm-ebpf"

cargo +nightly build \
  ${CARGO_FLAG} \
  -Z build-std=core \
  --target bpfel-unknown-none

OUT="target/bpfel-unknown-none/${PROFILE}/nsm-ebpf"
if [[ ! -f "$OUT" ]]; then
  echo "build did not produce $OUT" >&2
  exit 1
fi

echo "built: xdp/nsm-ebpf/${OUT}"
echo "run nsm with:  sudo ./target/release/nsm --interface <iface> --xdp --xdp-obj xdp/nsm-ebpf/${OUT}"
