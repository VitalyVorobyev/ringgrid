//! Compositional target model.
//!
//! A printed calibration target is described by [`TargetLayout`], which
//! composes four orthogonal aspects:
//!
//! - **lattice** ([`LatticeGeometry`]): hex or rectangular cell arrangement,
//! - **marker** ([`RingGeometry`]): the ring radii shared by every marker,
//! - **coding** ([`MarkerCoding`]): 16-sector coded rings or plain annuli,
//! - **fiducials** ([`OriginFiducials`]): optional origin/orientation dots.
//!
//! Targets round-trip through the `ringgrid.target.v5` JSON schema; the
//! legacy flat `ringgrid.target.v4` schema is auto-migrated on load.

mod error;
mod fiducials;
mod lattice;
pub(crate) mod layout;
mod ring;
mod schema;

pub use error::{TargetLoadError, TargetValidationError};
pub use fiducials::OriginFiducials;
pub use lattice::{HexGeometry, LatticeGeometry, RectGeometry};
pub use layout::{TargetCell, TargetLayout};
pub use ring::{CodedRingSpec, MarkerCoding, RingGeometry};
