# vcpkg overlay port for the ringgrid C ABI.
#
# ringgrid-c is built by cargo, so a Rust toolchain (cargo on PATH) is required
# at build time — vcpkg does not provide one. Install with:
#
#   vcpkg install ringgrid --overlay-ports=crates/ringgrid-c/vcpkg
#
# For local iteration against a working checkout (no published tag needed), set
# RINGGRID_SOURCE_DIR to the repository root; otherwise the port fetches the
# released tag from GitHub.

find_program(RINGGRID_CARGO NAMES cargo)
if(NOT RINGGRID_CARGO)
    message(FATAL_ERROR
        "cargo not found. The ringgrid port builds from Rust source and needs a "
        "Rust toolchain on PATH (https://rustup.rs).")
endif()

if(DEFINED ENV{RINGGRID_SOURCE_DIR})
    set(SOURCE_PATH "$ENV{RINGGRID_SOURCE_DIR}")
    message(STATUS "ringgrid: building from local checkout ${SOURCE_PATH}")
else()
    vcpkg_from_github(
        OUT_SOURCE_PATH SOURCE_PATH
        REPO VitalyVorobyev/ringgrid
        REF "v${VERSION}"
        # Replace with the release tarball SHA512 when tagging (vcpkg prints the
        # expected value on first fetch with SHA512 0).
        SHA512 0
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
