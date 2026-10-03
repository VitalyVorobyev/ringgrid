//! Workspace task runner.
//!
//! `cargo xtask emit-schemas [--check]` writes the JSON Schemas for the
//! user-facing ringgrid configs into `schemas/`. They ship inside the
//! `@vitavision/ringgrid` npm package so a schema-driven form can edit the
//! JSON the wasm API accepts. With `--check` nothing is written: the command
//! fails when the committed files differ from what the current source generates.

use std::error::Error;
use std::path::{Path, PathBuf};

mod emit_schemas;

const USAGE: &str = "usage: cargo xtask <command>\n\ncommands:\n  emit-schemas [--check]";

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        return Err(USAGE.into());
    };
    match cmd.as_str() {
        "emit-schemas" => {
            let mut check = false;
            for arg in args {
                match arg.as_str() {
                    "--check" => check = true,
                    other => return Err(format!("unknown argument `{other}`\n{USAGE}").into()),
                }
            }
            emit_schemas::run(&workspace_root()?, check)
        }
        other => Err(format!("unknown xtask `{other}`\n{USAGE}").into()),
    }
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    // CARGO_MANIFEST_DIR points at xtask/, so the workspace root is its parent.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map_err(|_| "CARGO_MANIFEST_DIR is unset; run via `cargo xtask`")?;
    Path::new(&manifest_dir)
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask/ has no parent directory".into())
}
