/*
 * ringgrid C ABI — scaffold header.
 *
 * Hand-written to match the current scaffold surface so C/C++ consumers can use
 * it today. Once cbindgen is wired in it will be regenerated from src/lib.rs:
 *   cbindgen --config cbindgen.toml --output include/ringgrid.h .
 *
 * Ownership: every returned `char *` is heap-owned by the caller and must be
 * released with ringgrid_string_free(). A NULL return signals an error.
 */
#ifndef RINGGRID_H
#define RINGGRID_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ringgrid version string. Free with ringgrid_string_free(). */
char *ringgrid_version(void);

/* Default coded-hex target as ringgrid.target.v5 JSON. Free with ringgrid_string_free(). */
char *ringgrid_default_target_json(void);

/* 24x24 plain-rect target (with origin dots) as JSON. Free with ringgrid_string_free(). */
char *ringgrid_rect_24x24_target_json(void);

/*
 * Detect markers in an 8-bit grayscale image.
 *
 * target_json: NUL-terminated ringgrid.target.v5 (or legacy v4) JSON.
 * pixels:      width * height row-major grayscale bytes.
 * Returns a DetectionResult JSON string, or NULL on error. Free with
 * ringgrid_string_free().
 */
char *ringgrid_detect_gray(const char *target_json,
                           const uint8_t *pixels,
                           uint32_t width,
                           uint32_t height);

/* Free a string returned by any ringgrid_* function (NULL is a no-op). */
void ringgrid_string_free(char *s);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* RINGGRID_H */
