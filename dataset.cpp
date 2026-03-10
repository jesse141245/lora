#include "dataset.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace {

    constexpr int kDefaultIgnoreIndex = -100;

    static std::vector<int> parseIntVector(
        const nlohmann::json& j,
        const char* fieldName)
    {
        if (!j.contains(fieldName)) {
            throw std::runtime_error(std::string("Dataset: missing field '") + fieldName + "'");
        }
        if (!j[fieldName].is_array()) {
            throw std::runtime_error(std::string("Dataset: field '") + fieldName + "' must be an array");
        }

        std::vector<int> out;
        out.reserve(j[fieldName].size());

        for (const auto& v : j[fieldName]) {
            if (!v.is_number_integer()) {
                throw std::runtime_error(std::string("Dataset: field '") + fieldName + "' must contain integers");
            }
            out.push_back(v.get<int>());
        }

        return out;
    }

    static std::vector<int> buildDefaultTargets(const std::vector<int>& inputIds)
    {
        std::vector<int> targetIds(inputIds.size(), kDefaultIgnoreIndex);
        if (inputIds.empty()) return targetIds;

        for (std::size_t i = 0; i + 1 < inputIds.size(); ++i) {
            targetIds[i] = inputIds[i + 1];
        }
        targetIds.back() = kDefaultIgnoreIndex;
        return targetIds;
    }

    static std::vector<int> buildDefaultAttentionMask(std::size_t n)
    {
        return std::vector<int>(n, 1);
    }

    static TrainingSample parseSampleObject(const nlohmann::json& j)
    {
        if (!j.is_object()) {
            throw std::runtime_error("Dataset: each sample must be a JSON object");
        }

        TrainingSample s;
        s.inputIds = parseIntVector(j, "input_ids");

        if (s.inputIds.empty()) {
            throw std::runtime_error("Dataset: input_ids cannot be empty");
        }

        if (j.contains("target_ids")) {
            if (!j["target_ids"].is_array()) {
                throw std::runtime_error("Dataset: field 'target_ids' must be an array");
            }
            s.targetIds.reserve(j["target_ids"].size());
            for (const auto& v : j["target_ids"]) {
                if (!v.is_number_integer()) {
                    throw std::runtime_error("Dataset: field 'target_ids' must contain integers");
                }
                s.targetIds.push_back(v.get<int>());
            }
        }
        else {
            s.targetIds = buildDefaultTargets(s.inputIds);
        }

        if (j.contains("attention_mask")) {
            if (!j["attention_mask"].is_array()) {
                throw std::runtime_error("Dataset: field 'attention_mask' must be an array");
            }
            s.attentionMask.reserve(j["attention_mask"].size());
            for (const auto& v : j["attention_mask"]) {
                if (!v.is_number_integer()) {
                    throw std::runtime_error("Dataset: field 'attention_mask' must contain integers");
                }
                const int x = v.get<int>();
                if (x != 0 && x != 1) {
                    throw std::runtime_error("Dataset: attention_mask values must be 0 or 1");
                }
                s.attentionMask.push_back(x);
            }
        }
        else {
            s.attentionMask = buildDefaultAttentionMask(s.inputIds.size());
        }

        if (s.targetIds.size() != s.inputIds.size()) {
            throw std::runtime_error("Dataset: target_ids size must match input_ids size");
        }
        if (s.attentionMask.size() != s.inputIds.size()) {
            throw std::runtime_error("Dataset: attention_mask size must match input_ids size");
        }

        return s;
    }

    static bool looksLikeJsonArrayFile(const std::string& text)
    {
        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) continue;
            return c == '[';
        }
        return false;
    }

    static std::string readWholeFile(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Dataset: failed to open file: " + path);
        }

        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    static void truncateSampleInPlace(TrainingSample& s, std::size_t maxSeqLen)
    {
        if (maxSeqLen == 0) {
            throw std::runtime_error("Dataset: maxSeqLen must be > 0");
        }

        if (s.inputIds.size() <= maxSeqLen) return;

        s.inputIds.resize(maxSeqLen);
        s.targetIds.resize(maxSeqLen);
        s.attentionMask.resize(maxSeqLen);

        if (!s.targetIds.empty() && s.attentionMask.back() != 0) {
            s.targetIds.back() = kDefaultIgnoreIndex;
        }
    }

    static void padSampleInPlace(
        TrainingSample& s,
        std::size_t maxSeqLen,
        int ignoreIndex)
    {
        if (s.inputIds.size() > maxSeqLen) {
            throw std::runtime_error("Dataset: internal error, pad called on oversized sample");
        }

        if (s.inputIds.size() == maxSeqLen) return;

        s.inputIds.resize(maxSeqLen, 0);
        s.targetIds.resize(maxSeqLen, ignoreIndex);
        s.attentionMask.resize(maxSeqLen, 0);
    }

    static void normalizeSampleInPlace(
        TrainingSample& s,
        const TrainerConfig& cfg)
    {
        truncateSampleInPlace(s, cfg.maxSeqLen);
        padSampleInPlace(s, cfg.maxSeqLen, cfg.ignoreIndex);

        if (s.inputIds.size() != cfg.maxSeqLen ||
            s.targetIds.size() != cfg.maxSeqLen ||
            s.attentionMask.size() != cfg.maxSeqLen) {
            throw std::runtime_error("Dataset: normalization failed to produce maxSeqLen-sized sample");
        }

        for (std::size_t i = 0; i < s.attentionMask.size(); ++i) {
            if (s.attentionMask[i] == 0) {
                s.targetIds[i] = cfg.ignoreIndex;
            }
        }

        for (std::size_t i = s.attentionMask.size(); i > 0; --i) {
            const std::size_t idx = i - 1;
            if (s.attentionMask[idx] != 0) {
                s.targetIds[idx] = cfg.ignoreIndex;
                break;
            }
        }
    }

} // namespace

Dataset::Dataset(const std::string& path, const TrainerConfig& cfg)
{
    if (path.empty()) {
        throw std::runtime_error("Dataset: path is empty");
    }
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Dataset: file does not exist: " + path);
    }
    if (cfg.maxSeqLen == 0) {
        throw std::runtime_error("Dataset: cfg.maxSeqLen must be > 0");
    }

    const std::string text = readWholeFile(path);
    if (text.empty()) {
        throw std::runtime_error("Dataset: file is empty: " + path);
    }

    samples_.clear();

    std::size_t rawMinLen = std::numeric_limits<std::size_t>::max();
    std::size_t rawMaxLen = 0;
    std::uint64_t rawTotalTokens = 0;
    std::size_t truncatedCount = 0;
    std::size_t paddedCount = 0;

    auto pushParsedSample = [&](TrainingSample s) {
        const std::size_t rawLen = s.inputIds.size();
        rawMinLen = std::min(rawMinLen, rawLen);
        rawMaxLen = std::max(rawMaxLen, rawLen);
        rawTotalTokens += static_cast<std::uint64_t>(rawLen);

        if (rawLen > cfg.maxSeqLen) ++truncatedCount;
        if (rawLen < cfg.maxSeqLen) ++paddedCount;

        normalizeSampleInPlace(s, cfg);
        samples_.push_back(std::move(s));
        };

    if (looksLikeJsonArrayFile(text)) {
        nlohmann::json root = nlohmann::json::parse(text);

        if (!root.is_array()) {
            throw std::runtime_error("Dataset: top-level JSON must be an array");
        }

        samples_.reserve(root.size());
        for (const auto& item : root) {
            pushParsedSample(parseSampleObject(item));
        }
    }
    else {
        std::istringstream in(text);
        std::string line;
        std::size_t lineNo = 0;

        while (std::getline(in, line)) {
            ++lineNo;

            bool allWs = true;
            for (char c : line) {
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    allWs = false;
                    break;
                }
            }
            if (allWs) continue;

            try {
                const auto j = nlohmann::json::parse(line);
                pushParsedSample(parseSampleObject(j));
            }
            catch (const std::exception& ex) {
                throw std::runtime_error(
                    "Dataset: failed to parse JSONL line " +
                    std::to_string(lineNo) + ": " + ex.what()
                );
            }
        }
    }

    if (samples_.empty()) {
        throw std::runtime_error("Dataset: no samples loaded from " + path);
    }

    std::cout << "Loaded dataset: " << path
        << " | samples=" << samples_.size()
        << " | rawMinLen=" << rawMinLen
        << " | rawMaxLen=" << rawMaxLen
        << " | rawAvgLen=" << (samples_.empty() ? 0.0
            : static_cast<double>(rawTotalTokens) / static_cast<double>(samples_.size()))
        << " | truncated=" << truncatedCount
        << " | padded=" << paddedCount
        << " | finalLen=" << cfg.maxSeqLen
        << '\n';
}

std::size_t Dataset::size() const noexcept {
    return samples_.size();
}

const TrainingSample& Dataset::get(std::size_t idx) const {
    return samples_.at(idx);
}