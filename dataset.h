#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "config.h"

struct TrainingSample {
    std::vector<int> inputIds;
    std::vector<int> targetIds;
    std::vector<int> attentionMask; // 1 = real token, 0 = pad
};

class Dataset {
public:
    Dataset(const std::string& path, const TrainerConfig& cfg);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const TrainingSample& get(std::size_t idx) const;

private:
    std::vector<TrainingSample> samples_;
};