# vcpkg overlay port for the ringgrid C ABI.
#
# ringgrid-c is built by cargo, so a Rust toolchain (cargo on PATH) is required
# at build time — vcpkg does not provide one.
#
# Two source modes:
#
#   1. Local checkout (the supported path until ringgrid is in the upstream
#      vcpkg registry — no published tag or archive hash needed):
#
#        RINGGRID_SOURCE_DIR=/path/to/ringgrid \
#          vcpkg install ringgrid --overlay-ports=crates/ringgrid-c/vcpkg
#
#   2. Released GitHub tarball (RINGGRID_SOURCE_DIR unset): fetches tag
#      v${VERSION}. This path only works once a release is tagged AND its real
#      SHA512 is committed below — see the `vcpkg_from_github` block.

# vcpkg builds ports in a sanitized environment that drops rustup's
# ~/.cargo/bin from PATH, so probe the standard rustup install locations too.
# USERPROFILE/HOME are passed through by vcpkg by default; keep CARGO_HOME with
# `--x-keep-env-vars` / VCPKG_KEEP_ENV_VARS if it lives in a custom location.
find_program(RINGGRID_CARGO NAMES cargo
    PATHS
        "$ENV{CARGO_HOME}/bin"
        "$ENV{USERPROFILE}/.cargo/bin"
        "$ENV{HOME}/.cargo/bin")
if(NOT RINGGRID_CARGO)
    message(FATAL_ERROR
        "cargo not found. The ringgrid port builds from Rust source and needs a "
        "Rust toolchain (https://rustup.rs); install it or set CARGO_HOME.")
endif()

if(DEFINED ENV{RINGGRID_SOURCE_DIR})
    set(SOURCE_PATH "$ENV{RINGGRID_SOURCE_DIR}")
    message(STATUS "ringgrid: building from local checkout ${SOURCE_PATH}")
else()
    vcpkg_from_github(
        OUT_SOURCE_PATH SOURCE_PATH
        REPO VitalyVorobyev/ringgrid
        REF "v${VERSION}"
        # Tarball SHA512 for the tagged release. RELEASE STEP: regenerate on
        # every version bump — `vcpkg install ringgrid --overlay-ports=...`
        # prints the "Actual hash", or `shasum -a 512` the
        # github.com/VitalyVorobyev/ringgrid/archive/v<version>.tar.gz tarball.
        # (Set to 0 to intentionally break mode 2 and print the expected hash.)
        SHA512 a9ae42faa46eb04419caff59f8c3c50bf8aed0acbe1e5a7b18daa09a13a36640a81de0786b7c0024113fd11eb5628e42211bfe592eb15ecd36ce1972c50987ad
        HEAD_REF main
    )
endif()

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" RINGGRID_BUILD_SHARED)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/crates/ringgrid-c"
    OPTIONS
        "-DRINGGRID_BUILD_SHARED=${RINGGRID_BUILD_SHARED}"
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME ringgrid CONFIG_PATH lib/cmake/ringgrid)
vcpkg_fixup_pkgconfig()

file(INSTALL "${SOURCE_PATH}/crates/ringgrid-c/vcpkg/usage"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST
    "${SOURCE_PATH}/LICENSE-MIT"
    "${SOURCE_PATH}/LICENSE-APACHE")
