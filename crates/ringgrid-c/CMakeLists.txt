# CMake packaging for the ringgrid C ABI.
#
# Builds the crate with cargo, installs the library + both headers + a CMake
# package config, so consumers do:
#
#     find_package(ringgrid CONFIG REQUIRED)
#     target_link_libraries(app PRIVATE ringgrid::ringgrid)
#
# Linkage follows RINGGRID_BUILD_SHARED (defaulting to BUILD_SHARED_LIBS); vcpkg
# maps VCPKG_LIBRARY_LINKAGE onto it.
cmake_minimum_required(VERSION 3.20)

project(ringgrid
    VERSION 0.10.1
    DESCRIPTION "C ABI for the ringgrid ring-target detector"
    LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(RINGGRID_BUILD_SHARED "Build the shared library (else static)" ${BUILD_SHARED_LIBS})

# Also probe rustup's standard install dirs: sandboxed builders (notably vcpkg)
# sanitize PATH and drop ~/.cargo/bin, so a bare PATH lookup misses cargo.
find_program(CARGO_EXECUTABLE cargo
    PATHS
        "$ENV{CARGO_HOME}/bin"
        "$ENV{USERPROFILE}/.cargo/bin"
        "$ENV{HOME}/.cargo/bin"
    DOC "Path to the cargo build tool (a Rust toolchain is required to build ringgrid)"
    REQUIRED)

# Cargo emits into an isolated target dir so it never pollutes the repo target/.
set(_rg_cargo_target "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(_rg_manifest "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml")

if(RINGGRID_BUILD_SHARED)
    set(_rg_libfile "${CMAKE_SHARED_LIBRARY_PREFIX}ringgrid_c${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_rg_kind SHARED)
else()
    set(_rg_libfile "${CMAKE_STATIC_LIBRARY_PREFIX}ringgrid_c${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(_rg_kind STATIC)
endif()
set(_rg_libpath "${_rg_cargo_target}/release/${_rg_libfile}")

# Windows MSVC: a cdylib produces ringgrid_c.dll plus a separate import lib
# ringgrid_c.dll.lib to link against.
set(_rg_implib_name "")
if(RINGGRID_BUILD_SHARED AND MSVC)
    set(_rg_implib "${_rg_cargo_target}/release/ringgrid_c.dll.lib")
    set(_rg_implib_name "ringgrid_c.dll.lib")
endif()

add_custom_command(
    OUTPUT "${_rg_libpath}"
    COMMAND "${CARGO_EXECUTABLE}" build --release
            --manifest-path "${_rg_manifest}"
            --target-dir "${_rg_cargo_target}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building ringgrid_c (${_rg_kind}) with cargo"
    VERBATIM)
add_custom_target(ringgrid_cargo ALL DEPENDS "${_rg_libpath}")

# Transitive system libraries that the Rust std staticlib needs. Sourced from
# `cargo rustc -- --print native-static-libs`. Only needed for static linkage.
set(RINGGRID_SYSTEM_LIBS "")
if(NOT RINGGRID_BUILD_SHARED)
    if(APPLE)
        set(RINGGRID_SYSTEM_LIBS iconv "-framework CoreFoundation" "-framework Security")
    elseif(WIN32)
        set(RINGGRID_SYSTEM_LIBS ws2_32 userenv ntdll bcrypt advapi32 kernel32)
    else()
        set(RINGGRID_SYSTEM_LIBS pthread dl m)
    endif()
endif()

add_library(ringgrid_imported ${_rg_kind} IMPORTED GLOBAL)
add_dependencies(ringgrid_imported ringgrid_cargo)
set_target_properties(ringgrid_imported PROPERTIES
    IMPORTED_LOCATION "${_rg_libpath}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(DEFINED _rg_implib)
    set_target_properties(ringgrid_imported PROPERTIES IMPORTED_IMPLIB "${_rg_implib}")
endif()
if(RINGGRID_SYSTEM_LIBS)
    set_target_properties(ringgrid_imported PROPERTIES
        INTERFACE_LINK_LIBRARIES "${RINGGRID_SYSTEM_LIBS}")
endif()
add_library(ringgrid::ringgrid ALIAS ringgrid_imported)

# ── Install ─────────────────────────────────────────────────────────

if(RINGGRID_BUILD_SHARED AND WIN32)
    # Windows shared build: `_rg_libpath` is the DLL — a runtime artifact that
    # belongs in bin, not lib. Only the import library (`_rg_implib`) goes in lib
    # so the standard Windows/vcpkg layout holds (lib/ = link-time, bin/ = runtime).
    install(FILES "${_rg_cargo_target}/release/ringgrid_c.dll"
            DESTINATION "${CMAKE_INSTALL_BINDIR}")
else()
    # Static archive, or a non-Windows shared object: `_rg_libpath` is the
    # linkable artifact and belongs in lib.
    install(FILES "${_rg_libpath}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()
if(DEFINED _rg_implib)
    install(FILES "${_rg_implib}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ringgrid.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/ringgrid.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# CMake package config: consumers `find_package(ringgrid CONFIG REQUIRED)`.
set(RINGGRID_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/ringgrid")
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ringgrid-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ringgrid-config.cmake"
    INSTALL_DESTINATION "${RINGGRID_INSTALL_CMAKEDIR}"
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_BINDIR CMAKE_INSTALL_INCLUDEDIR)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ringgrid-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ringgrid-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ringgrid-config-version.cmake"
    DESTINATION "${RINGGRID_INSTALL_CMAKEDIR}")

# pkg-config for non-CMake consumers.
set(RINGGRID_PC_LIBS_PRIVATE "")
foreach(_lib IN LISTS RINGGRID_SYSTEM_LIBS)
    string(FIND "${_lib}" "-framework" _is_framework)
    if(_is_framework GREATER -1)
        string(APPEND RINGGRID_PC_LIBS_PRIVATE " ${_lib}")
    else()
        string(APPEND RINGGRID_PC_LIBS_PRIVATE " -l${_lib}")
    endif()
endforeach()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ringgrid.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ringgrid.pc"
    @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ringgrid.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
