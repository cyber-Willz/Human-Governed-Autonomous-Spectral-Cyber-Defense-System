#!/usr/bin/env sh
# Fetch the 66-region human structural connectome (diffusion-MRI group average,
# Desikan-Killiany-style labels) distributed with The Virtual Brain's tvb-data repo.
# Check the tvb-data documentation for provenance, licence and how to cite it.
#
#   sh scripts/fetch_tvb66.sh [dest]     # default dest: data/tvb66
#   connectome_rnn train --tvb data/tvb66
set -eu
dest="${1:-data/tvb66}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$dest"
curl -fsSL -o "$tmp/connectivity_66.zip" \
  "https://github.com/the-virtual-brain/tvb-data/raw/master/tvb_data/connectivity/connectivity_66.zip"
unzip -o -q "$tmp/connectivity_66.zip" -d "$dest"
echo "connectome written to $dest:"
ls "$dest"
