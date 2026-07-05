//! Internal support for the `ringgrid` command-line tool (built with
//! `--features cli`).
//!
//! **Not part of the stable library API.** These items back the published
//! `ringgrid` binary and are reused by the in-repo dev CLI; the module is
//! `#[doc(hidden)]` and may change without a semver bump. Library users should
//! depend on [`Detector`](crate::Detector) and [`TargetLayout`](crate::TargetLayout)
//! directly.
#![allow(missing_docs)]

mod artifacts;
mod detect;
mod recipe;

pub use artifacts::{ArtifactError, write_target_artifacts};
pub use detect::{
    BatchOutcome, DetectRunError, build_detector, decoded_count, detect_file, detect_image,
    image_files_in_dir, load_target, run_batch,
};
pub use recipe::{
    CodingRecipe, FiducialMode, FiducialsRecipe, Format, LatticeRecipe, MarkerRecipe, RecipeError,
    RenderRecipe, TargetRecipe,
};

/// Built-in example recipes, embedded so `cargo install`ed users get them
/// without a repository checkout. Ordered as the 6 valid target combinations.
pub const EXAMPLE_RECIPES: &[(&str, &str)] = &[
    (
        "hex_coded",
        include_str!("../../examples/targets/hex_coded.toml"),
    ),
    (
        "rect_coded",
        include_str!("../../examples/targets/rect_coded.toml"),
    ),
    (
        "hex_plain_dots",
        include_str!("../../examples/targets/hex_plain_dots.toml"),
    ),
    (
        "hex_plain_nodots",
        include_str!("../../examples/targets/hex_plain_nodots.toml"),
    ),
    (
        "rect_plain_dots",
        include_str!("../../examples/targets/rect_plain_dots.toml"),
    ),
    (
        "rect_plain_nodots",
        include_str!("../../examples/targets/rect_plain_nodots.toml"),
    ),
];

/// Look up a built-in example recipe by name.
pub fn example_recipe(name: &str) -> Option<&'static str> {
    EXAMPLE_RECIPES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, text)| *text)
}
