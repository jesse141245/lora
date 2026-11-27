#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>

namespace envutil {

inline std::optional<std::string> tryGetEnvString(const char* name) {
#if defined(_WIN32)
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || !value || len == 0 || value[0] == '\0') {
        if (value) {
            std::free(value);
        }
        return std::nullopt;
    }
    std::string out(value);
    std::free(value);
    return out;
#else
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

inline std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline bool envStringTruthy(const std::string& s) {
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

inline bool envStringFalsy(const std::string& s) {
    return s == "0" || s == "false" || s == "no" || s == "off";
}

inline std::optional<bool> envBoolOverride(const char* name) {
    const auto value = tryGetEnvString(name);
    if (!value) {
        return std::nullopt;
    }
    const std::string lowered = lowerCopy(*value);
    if (envStringTruthy(lowered)) return true;
    if (envStringFalsy(lowered)) return false;
    return std::nullopt;
}

inline bool envBoolOr(const char* name, bool fallback) {
    const auto value = tryGetEnvString(name);
    if (!value) {
        return fallback;
    }
    const std::string lowered = lowerCopy(*value);
    if (envStringTruthy(lowered)) return true;
    if (envStringFalsy(lowered)) return false;
    return fallback;
}

inline std::optional<std::size_t> envSizeTOverride(const char* name) {
    const auto value = tryGetEnvString(name);
    if (!value) return std::nullopt;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value->c_str(), &end, 10);
    if (end == value->c_str() || parsed == 0) return std::nullopt;
    return static_cast<std::size_t>(parsed);
}

inline std::size_t envSizeTOr(const char* name, std::size_t fallback) {
    const auto value = tryGetEnvString(name);
    if (!value) return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value->c_str(), &end, 10);
    if (end == value->c_str() || parsed == 0) return fallback;
    return static_cast<std::size_t>(parsed);
}

} // namespace envutil
