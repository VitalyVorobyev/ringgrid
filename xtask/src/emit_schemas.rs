//! Generate (or verify) the JSON Schemas in `schemas/`.

use std::error::Error;
use std::path::{Path, PathBuf};

use schemars::schema_for;
use serde_json::Value;

/// Output file name, schema.
fn schemas() -> Vec<(&'static str, Value)> {
    vec![
        (
            "detect_config.json",
            to_value(schema_for!(ringgrid::DetectConfig)),
        ),
        ("target_spec.json", to_value(ringgrid::target_spec_schema())),
        (
            "target_render_options.json",
            to_value(schema_for!(ringgrid::TargetRenderOptions)),
        ),
    ]
}

fn to_value(schema: schemars::Schema) -> Value {
    let mut value = schema.to_value();
    postprocess(&mut value);
    value
}

pub fn run(workspace_root: &Path, check: bool) -> Result<(), Box<dyn Error>> {
    let out_dir = workspace_root.join("schemas");
    let entries = schemas();

    if check {
        let mut drift: Vec<(PathBuf, &'static str)> = Vec::new();
        for (name, schema) in &entries {
            let path = out_dir.join(name);
            let expected = render(schema)?;
            match std::fs::read_to_string(&path) {
                Err(_) => drift.push((path, "missing")),
                Ok(on_disk) if on_disk == expected => {}
                // Same bytes once CRLF is normalised: a checkout problem
                // (`.gitattributes` forces LF), not a stale schema.
                Ok(on_disk) if on_disk.replace("\r\n", "\n") == expected => {
                    drift.push((path, "CRLF line endings on disk"));
                }
                Ok(_) => drift.push((path, "out of date")),
            }
        }
        if drift.is_empty() {
            println!("schemas up to date ({} files)", entries.len());
            return Ok(());
        }
        for (path, why) in &drift {
            eprintln!("schema drift: {} ({why})", path.display());
        }
        return Err(format!(
            "{} schema file(s) differ from the source; run `cargo xtask emit-schemas` and commit the result",
            drift.len()
        )
        .into());
    }

    std::fs::create_dir_all(&out_dir)?;
    for (name, schema) in &entries {
        std::fs::write(out_dir.join(name), render(schema)?)?;
    }
    println!("emitted {} schemas to {}", entries.len(), out_dir.display());
    Ok(())
}

/// Pretty JSON with a trailing newline.
fn render(schema: &Value) -> Result<String, Box<dyn Error>> {
    let mut text = serde_json::to_string_pretty(schema)?;
    text.push('\n');
    Ok(text)
}

/// Make the schemas friendlier to form libraries.
///
/// - Rustdoc intra-doc link syntax in `description` / `title` strings (the
///   strings come from Rust doc comments, but a form renders them as plain
///   text, where ``[`Foo`]`` and ``[`Foo`](path)`` are noise) is stripped.
/// - Rust-width `format` values (`float`, `uint32`, ...) are dropped. They are
///   not JSON Schema formats, and validators that treat unknown formats as an
///   error (Ajv in its default strict mode) would reject the whole schema. The
///   `type`, and the `minimum: 0` schemars adds to unsigned integers, remain.
/// - `default` numbers that are widened `f32` values (`0.35` serialized as
///   `0.3499999940395355`) are rewritten to their shortest `f32` spelling.
fn postprocess(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if map
                .get("format")
                .and_then(Value::as_str)
                .is_some_and(is_rust_numeric_format)
            {
                map.remove("format");
            }
            for (key, child) in map.iter_mut() {
                match (key.as_str(), child) {
                    ("description" | "title", Value::String(s)) => *s = strip_doc_links(s),
                    ("default", child) => shorten_f32_defaults(child),
                    (_, child) => postprocess(child),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(postprocess),
        _ => {}
    }
}

/// `default` values are produced by serializing the Rust defaults, and an `f32`
/// field widens to `f64` on the way into `serde_json::Value`. Rewrite any
/// number that is exactly an `f32` to that `f32`'s shortest decimal form.
fn shorten_f32_defaults(value: &mut Value) {
    match value {
        Value::Number(n) => {
            if let Some(x) = n.as_f64() {
                let narrow = x as f32;
                if f64::from(narrow) == x
                    && let Ok(short) = narrow.to_string().parse::<f64>()
                    && short != x
                    && let Some(short) = serde_json::Number::from_f64(short)
                {
                    *n = short;
                }
            }
        }
        Value::Object(map) => map.values_mut().for_each(shorten_f32_defaults),
        Value::Array(items) => items.iter_mut().for_each(shorten_f32_defaults),
        _ => {}
    }
}

fn is_rust_numeric_format(format: &str) -> bool {
    matches!(format, "float" | "double")
        || format
            .strip_prefix("uint")
            .or_else(|| format.strip_prefix("int"))
            .is_some_and(|bits| bits.chars().all(|c| c.is_ascii_digit()))
}

/// ``[`X`](target)`` and ``[`X`]`` become `` `X` ``. Plain text in brackets is
/// left alone.
fn strip_doc_links(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("[`") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        // `after` begins with the backtick; the code span ends at the next one.
        let Some(code_end) = after[1..].find('`').map(|i| i + 2) else {
            out.push_str(&rest[start..]);
            return out;
        };
        let code = &after[..code_end];
        let tail = &after[code_end..];
        if let Some(tail) = tail.strip_prefix(']') {
            out.push_str(&code.replacen("`crate::", "`", 1));
            rest = if let Some(link) = tail.strip_prefix('(') {
                link.find(')').map_or(tail, |i| &link[i + 1..])
            } else {
                tail
            };
        } else {
            out.push_str(&rest[start..start + 1 + code_end]);
            rest = tail;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::{is_rust_numeric_format, postprocess, shorten_f32_defaults, strip_doc_links};
    use serde_json::json;

    #[test]
    fn drops_only_rust_numeric_formats() {
        for f in ["float", "double", "uint", "uint8", "uint64", "int32", "int"] {
            assert!(is_rust_numeric_format(f), "{f}");
        }
        for f in ["date-time", "uri", "email", "uint-ish", ""] {
            assert!(!is_rust_numeric_format(f), "{f}");
        }
        let mut v = json!({
            "properties": { "a": { "type": "number", "format": "float" }, "b": { "type": "string", "format": "uri" } }
        });
        postprocess(&mut v);
        assert!(v["properties"]["a"].get("format").is_none());
        assert_eq!(v["properties"]["b"]["format"], "uri");
    }

    #[test]
    fn shortens_widened_f32_defaults_only() {
        let mut v = json!({
            "a": f64::from(0.35_f32), "b": 0.1_f64, "c": 6.0, "d": [f64::from(0.08_f32), 3],
            "e": {"x": 1e-6_f64, "y": null}
        });
        shorten_f32_defaults(&mut v);
        assert_eq!(v["a"], json!(0.35));
        assert_eq!(v["b"], json!(0.1));
        assert_eq!(v["c"], json!(6.0));
        assert_eq!(v["d"], json!([0.08, 3]));
        assert_eq!(v["e"]["x"], json!(1e-6));
    }

    #[test]
    fn strips_intra_doc_links() {
        assert_eq!(
            strip_doc_links("See [`Foo::bar`], [`crate::Qux`] and [`Baz`](super::Baz) now."),
            "See `Foo::bar`, `Qux` and `Baz` now."
        );
        assert_eq!(strip_doc_links("range [0, 1] stays"), "range [0, 1] stays");
        assert_eq!(strip_doc_links("[`unclosed"), "[`unclosed");
    }
}
