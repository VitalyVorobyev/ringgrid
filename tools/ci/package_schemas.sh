#!/usr/bin/env bash
# Ship the JSON Schemas in the npm package.
#
# wasm-pack generates `pkg/package.json` with a closed `files` list, so the
# schemas have to be copied in and added to it explicitly. Then `npm pack
# --dry-run` proves the tarball would really contain them.
#
# Usage: tools/ci/package_schemas.sh [pkg-dir]   (default: crates/ringgrid-wasm/pkg)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="${1:-$ROOT/crates/ringgrid-wasm/pkg}"
SCHEMAS=(detect_config.json target_spec.json target_render_options.json)

test -f "$PKG/package.json" || { echo "no package.json in $PKG; run wasm-pack first" >&2; exit 1; }

rm -rf "$PKG/schemas"
mkdir -p "$PKG/schemas"
for name in "${SCHEMAS[@]}"; do
  cp "$ROOT/schemas/$name" "$PKG/schemas/$name"
done

# Append "schemas" to `files`, preserving the entries wasm-pack wrote.
node -e '
const fs = require("fs");
const path = process.argv[1] + "/package.json";
const pkg = JSON.parse(fs.readFileSync(path, "utf8"));
const files = pkg.files || [];
if (!files.includes("schemas")) files.push("schemas");
pkg.files = files;
fs.writeFileSync(path, JSON.stringify(pkg, null, 2) + "\n");
' "$PKG"

# Assert the tarball would contain every schema file.
LISTING="$(cd "$PKG" && npm pack --dry-run --json)"
node -e '
const listing = JSON.parse(process.argv[1])[0].files.map((f) => f.path);
const missing = process.argv.slice(2).filter((n) => !listing.includes("schemas/" + n));
if (missing.length) {
  console.error("npm pack is missing: " + missing.join(", "));
  console.error("package contents: " + listing.join(", "));
  process.exit(1);
}
console.log("npm package ships " + (process.argv.length - 2) + " schema files");
' "$LISTING" "${SCHEMAS[@]}"
