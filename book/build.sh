#!/bin/bash
# Build the ringgrid book with the interactive WASM demo.
#
# Usage: ./book/build.sh
#
# Prerequisites:
#   - wasm-pack (cargo install wasm-pack)
#   - mdbook    (cargo install mdbook)
#
# The demo has a single canonical source in `book/demo/`. This script builds the
# WASM package and stages `book/demo/` (HTML/CSS/JS + sample images) together
# with the freshly built `pkg/` into `book/src/demo/` — a generated, gitignored
# directory — before rendering the book, so `/demo/` ships self-contained.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Building WASM package..."
wasm-pack build crates/ringgrid-wasm --target web --release

echo "==> Staging book/demo/ into book/src/demo/..."
rm -rf book/src/demo
mkdir -p book/src/demo/pkg book/src/demo/samples
cp book/demo/index.html book/demo/app.js book/demo/styles.css book/demo/samples.json book/demo/favicon.svg book/src/demo/
cp book/demo/samples/*                                                                book/src/demo/samples/
cp crates/ringgrid-wasm/pkg/ringgrid_wasm.js                                          book/src/demo/pkg/
cp crates/ringgrid-wasm/pkg/ringgrid_wasm_bg.wasm                                     book/src/demo/pkg/

echo "==> Building mdBook..."
mdbook build book

echo "==> Done. Output in book/book/ (demo at book/book/demo/)."
