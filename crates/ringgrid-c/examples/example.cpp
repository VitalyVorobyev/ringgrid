/*
 * example.cpp — minimal C++ consumer using the RAII convenience header.
 *
 * Demonstrates the ringgrid::Detector RAII wrapper, a std::string result, and
 * the exception path on invalid input.
 */
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ringgrid.hpp"

int main() {
    try {
        std::cout << "ringgrid " << ringgrid::version() << " (ABI " << ringgrid::abi_version()
                  << ")\n";

        ringgrid::Detector detector(ringgrid::default_target_json());

        const std::uint32_t width = 64, height = 48;
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height, 0);
        std::string result = detector.detect(pixels.data(), width, height);
        std::cout << "detect: " << result.size() << " bytes\n";
        if (result.find("detected_markers") == std::string::npos) {
            std::cerr << "missing detected_markers\n";
            return 1;
        }

        // The exception path: invalid target JSON must throw ringgrid::Error.
        try {
            ringgrid::Detector bad("not json");
            std::cerr << "expected an exception for invalid JSON\n";
            return 1;
        } catch (const ringgrid::Error& e) {
            std::cout << "expected error for invalid JSON: " << e.what() << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "unexpected error: " << e.what() << "\n";
        return 1;
    }
}
