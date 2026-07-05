/*
 * smoke.c — minimal C consumer of the ringgrid C ABI.
 *
 * Verifies the ABI version, queries the version string, builds a detector from
 * the default target, runs detection on an in-memory blank image, and frees
 * everything. No PNG decoder needed — the buffer is synthesized in memory.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ringgrid.h"

int main(void) {
    if (ringgrid_abi_version() != RINGGRID_ABI_VERSION) {
        fprintf(stderr, "ABI version mismatch\n");
        return 1;
    }

    char *version = NULL;
    if (ringgrid_version(&version) != RINGGRID_STATUS_OK) {
        return 1;
    }
    printf("ringgrid %s\n", version);
    ringgrid_string_free(version);

    char *target = NULL;
    RinggridStatus st = ringgrid_default_target_json(&target);
    if (st != RINGGRID_STATUS_OK) {
        fprintf(stderr, "target: %s\n", ringgrid_status_str(st));
        return 1;
    }

    RinggridDetector *det = NULL;
    st = ringgrid_detector_new(target, &det);
    ringgrid_string_free(target);
    if (st != RINGGRID_STATUS_OK) {
        fprintf(stderr, "detector_new: %s\n", ringgrid_status_str(st));
        return 1;
    }

    const uint32_t w = 64, h = 48;
    uint8_t *pixels = (uint8_t *)calloc((size_t)w * (size_t)h, 1);
    if (pixels == NULL) {
        ringgrid_detector_free(det);
        return 1;
    }

    char *result = NULL;
    st = ringgrid_detect(det, pixels, w, h, &result);
    free(pixels);
    if (st != RINGGRID_STATUS_OK) {
        fprintf(stderr, "detect: %s\n", ringgrid_status_str(st));
        ringgrid_detector_free(det);
        return 1;
    }

    int ok = strstr(result, "detected_markers") != NULL;
    printf("detect: %zu bytes, detected_markers present: %s\n", strlen(result),
           ok ? "yes" : "no");
    ringgrid_string_free(result);
    ringgrid_detector_free(det);

    return ok ? 0 : 1;
}
