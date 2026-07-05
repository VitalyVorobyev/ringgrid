/*
 * ringgrid.hpp — thin, header-only C++ convenience layer over the ringgrid C
 * ABI (ringgrid.h). RAII handle, std::string results, exceptions instead of
 * status codes, and an ABI-version guard. This is sugar over the C surface —
 * the C ABI in ringgrid.h remains the source of truth.
 *
 * Requires C++17.
 */
#ifndef RINGGRID_HPP
#define RINGGRID_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ringgrid.h"

namespace ringgrid {

/// Thrown when a C ABI call returns a non-OK status.
class Error : public std::runtime_error {
public:
    explicit Error(RinggridStatus status)
        : std::runtime_error(ringgrid_status_str(status)), status_(status) {}

    RinggridStatus status() const noexcept { return status_; }

private:
    RinggridStatus status_;
};

namespace detail {

inline void check(RinggridStatus status) {
    if (status != RINGGRID_STATUS_OK) {
        throw Error(status);
    }
}

/// Take ownership of a library-allocated string, always freeing it.
inline std::string take(char* owned) {
    if (owned == nullptr) {
        return std::string();
    }
    std::string out(owned);
    ringgrid_string_free(owned);
    return out;
}

/// Run a `char**`-out C call and return its owned payload as a std::string.
template <class F>
inline std::string owned_string(F&& call) {
    char* out = nullptr;
    check(call(&out));
    return take(out);
}

}  // namespace detail

// ── Free functions ──────────────────────────────────────────────────

/// Library version string.
inline std::string version() {
    return detail::owned_string([](char** out) { return ringgrid_version(out); });
}

/// Runtime ABI version of the linked library.
inline std::uint32_t abi_version() { return ringgrid_abi_version(); }

/// The default coded-hex target as `ringgrid.target.v5` JSON.
inline std::string default_target_json() {
    return detail::owned_string([](char** out) { return ringgrid_default_target_json(out); });
}

/// The 24×24 plain-rect target (with origin dots) as JSON.
inline std::string rect_24x24_target_json() {
    return detail::owned_string([](char** out) { return ringgrid_rect_24x24_target_json(out); });
}

/// The default detection config for a target, as JSON.
inline std::string default_config_json(const std::string& target_json) {
    return detail::owned_string(
        [&](char** out) { return ringgrid_default_config_json(target_json.c_str(), out); });
}

/// The four-tier-wide scale-tier preset as JSON.
inline std::string scale_tiers_four_tier_wide_json() {
    return detail::owned_string(
        [](char** out) { return ringgrid_scale_tiers_four_tier_wide_json(out); });
}

/// The two-tier-standard scale-tier preset as JSON.
inline std::string scale_tiers_two_tier_standard_json() {
    return detail::owned_string(
        [](char** out) { return ringgrid_scale_tiers_two_tier_standard_json(out); });
}

// ── Detector ────────────────────────────────────────────────────────

/// Owning, move-only RAII wrapper over a `RinggridDetector*`.
///
/// Detection methods take a raw grayscale/RGBA pixel pointer plus dimensions
/// (`std::vector<uint8_t>` users pass `data.data(), width, height`) and return
/// the JSON result as a `std::string`, throwing [`Error`] on failure.
class Detector {
public:
    /// Construct from a target layout JSON string (`v5`, or legacy `v4`).
    explicit Detector(const std::string& target_json) {
        check_abi();
        detail::check(ringgrid_detector_new(target_json.c_str(), &handle_));
    }

    static Detector with_marker_scale(const std::string& target_json, float min_px,
                                      float max_px) {
        check_abi();
        RinggridDetector* h = nullptr;
        detail::check(
            ringgrid_detector_with_marker_scale(target_json.c_str(), min_px, max_px, &h));
        return Detector(h);
    }

    static Detector with_marker_diameter(const std::string& target_json, float diameter_px) {
        check_abi();
        RinggridDetector* h = nullptr;
        detail::check(
            ringgrid_detector_with_marker_diameter(target_json.c_str(), diameter_px, &h));
        return Detector(h);
    }

    static Detector with_config(const std::string& target_json, const std::string& config_json) {
        check_abi();
        RinggridDetector* h = nullptr;
        detail::check(
            ringgrid_detector_with_config(target_json.c_str(), config_json.c_str(), &h));
        return Detector(h);
    }

    ~Detector() {
        if (handle_ != nullptr) {
            ringgrid_detector_free(handle_);
        }
    }

    Detector(Detector&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    Detector& operator=(Detector&& other) noexcept {
        if (this != &other) {
            if (handle_ != nullptr) {
                ringgrid_detector_free(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // ── Config ──────────────────────────────────────────────────────

    std::string config_json() const {
        return detail::owned_string(
            [&](char** out) { return ringgrid_detector_config_json(handle_, out); });
    }

    void update_config(const std::string& overlay_json) {
        detail::check(ringgrid_detector_update_config(handle_, overlay_json.c_str()));
    }

    // ── Detection ───────────────────────────────────────────────────

    std::string detect(const std::uint8_t* pixels, std::uint32_t width,
                       std::uint32_t height) const {
        return detail::owned_string(
            [&](char** out) { return ringgrid_detect(handle_, pixels, width, height, out); });
    }

    std::string detect_rgba(const std::uint8_t* pixels, std::uint32_t width,
                            std::uint32_t height) const {
        return detail::owned_string(
            [&](char** out) { return ringgrid_detect_rgba(handle_, pixels, width, height, out); });
    }

    std::string detect_with_diagnostics(const std::uint8_t* pixels, std::uint32_t width,
                                        std::uint32_t height) const {
        return detail::owned_string([&](char** out) {
            return ringgrid_detect_with_diagnostics(handle_, pixels, width, height, out);
        });
    }

    std::string detect_adaptive(const std::uint8_t* pixels, std::uint32_t width,
                                std::uint32_t height) const {
        return detail::owned_string([&](char** out) {
            return ringgrid_detect_adaptive(handle_, pixels, width, height, out);
        });
    }

    std::string detect_adaptive_with_hint(const std::uint8_t* pixels, std::uint32_t width,
                                          std::uint32_t height, float nominal_diameter_px) const {
        return detail::owned_string([&](char** out) {
            return ringgrid_detect_adaptive_with_hint(handle_, pixels, width, height,
                                                      nominal_diameter_px, out);
        });
    }

    std::string detect_multiscale(const std::uint8_t* pixels, std::uint32_t width,
                                  std::uint32_t height, const std::string& tiers_json) const {
        return detail::owned_string([&](char** out) {
            return ringgrid_detect_multiscale(handle_, pixels, width, height, tiers_json.c_str(),
                                              out);
        });
    }

    std::string detect_with_mapper(const std::uint8_t* pixels, std::uint32_t width,
                                   std::uint32_t height, const std::string& mapper_json) const {
        return detail::owned_string([&](char** out) {
            return ringgrid_detect_with_mapper(handle_, pixels, width, height, mapper_json.c_str(),
                                               out);
        });
    }

    std::string detect_with_mapper_diagnostics(const std::uint8_t* pixels, std::uint32_t width,
                                               std::uint32_t height,
                                               const std::string& mapper_json) const {
        return detail::owned_string([&](char** out) {
            return ringgrid_detect_with_mapper_diagnostics(handle_, pixels, width, height,
                                                           mapper_json.c_str(), out);
        });
    }

    // ── Proposals + heatmap ─────────────────────────────────────────

    std::string propose_with_heatmap(const std::uint8_t* pixels, std::uint32_t width,
                                     std::uint32_t height) {
        return detail::owned_string([&](char** out) {
            return ringgrid_propose_with_heatmap(handle_, pixels, width, height, out);
        });
    }

    /// The last heatmap, copied out of the handle's borrowed buffer.
    std::vector<float> heatmap() const {
        const float* ptr = nullptr;
        std::size_t len = 0;
        detail::check(ringgrid_heatmap_data(handle_, &ptr, &len));
        return std::vector<float>(ptr, ptr + len);
    }

    std::uint32_t heatmap_width() const { return ringgrid_heatmap_width(handle_); }
    std::uint32_t heatmap_height() const { return ringgrid_heatmap_height(handle_); }

    /// The underlying C handle (borrowed; still owned by this object).
    RinggridDetector* raw() const noexcept { return handle_; }

private:
    explicit Detector(RinggridDetector* handle) : handle_(handle) {}

    static void check_abi() {
        if (ringgrid_abi_version() != RINGGRID_ABI_VERSION) {
            throw std::runtime_error(
                "ringgrid ABI version mismatch between ringgrid.h and the linked library");
        }
    }

    RinggridDetector* handle_ = nullptr;
};

}  // namespace ringgrid

#endif  // RINGGRID_HPP
