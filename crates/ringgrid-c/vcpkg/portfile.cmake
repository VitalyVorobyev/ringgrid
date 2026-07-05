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
        # RELEASE STEP: after tagging v${VERSION}, replace the `0` placeholder
        # with the real tarball SHA512 (run `vcpkg install ringgrid
        # --overlay-ports=...` once and copy the "Actual hash" it prints, or
        # `vcpkg hash <downloaded.tar.gz>`). Until then, mode 2 above fails by
        # design — use RINGGRID_SOURCE_DIR (mode 1) for a source install.
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
