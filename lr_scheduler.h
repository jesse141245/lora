#pragma once
#include <cmath>
#include <algorithm>

// ===========================================================================
// LRScheduler — linear warmup + cosine decay
//
// Phase 1 (steps 0..warmupSteps-1): LR ramps linearly from 0 → peakLr.
//   This is intentionally SHORT (default 10% of maxSteps).
//   For LoRA cold-start: B must grow to A-comparable magnitude before A's
//   gradient activates. Full peakLr is needed immediately, so keep warmup
//   short (≤20 steps). The warmup just prevents a large step-1 spike.
//
// Phase 2 (steps warmupSteps..decayStartStep-1): optional hold at peakLr.
//   This preserves aggressive early adaptation on tiny overfit workloads.
//
// Phase 3 (steps decayStartStep..maxSteps): cosine decay from peakLr → minLr.
//   Smoothly reduces LR so the optimizer settles rather than oscillates.
//
// Usage in run() loop — call once per optimizer step:
//
//   scheduler.step(globalStep);                    // updates internal LR
//   optimizer_.setLr(scheduler.currentLr());       // applies to optimizer
// ===========================================================================

class LRScheduler {
public:
    LRScheduler() = default;

    LRScheduler(float peakLr, float minLr, int maxSteps, int warmupSteps, int decayStartStep = -1, int decaySteps = -1)
        : peakLr_{ peakLr }
        , minLr_{ minLr }
        , maxSteps_{ maxSteps }
        , warmupSteps_{ warmupSteps }
        , decayStartStep_{ decayStartStep }
        , decaySteps_{ decaySteps }
        , currentLr_{ 0.0f }
    {
    }

    // Call once per optimizer step with the 1-based step counter.
    void step(int globalStep) {
        if (maxSteps_ <= 0) { currentLr_ = peakLr_; return; }

        if (globalStep < warmupSteps_) {
            // Linear warmup: step 0 → tiny, step warmupSteps_ → peakLr_
            const float t = static_cast<float>(globalStep + 1)
                / static_cast<float>(warmupSteps_);
            currentLr_ = peakLr_ * t;
        }
        else if (decayStartStep_ <= warmupSteps_ || globalStep < decayStartStep_) {
            currentLr_ = peakLr_;
        }
        else {
            // Cosine decay from peakLr_ → minLr_
            const int effectiveDecaySteps =
                (decaySteps_ > 0)
                ? decaySteps_
                : std::max(1, maxSteps_ - decayStartStep_);
            const float progress =
                static_cast<float>(globalStep - decayStartStep_)
                / static_cast<float>(effectiveDecaySteps);
            const float clampedProgress = std::max(0.0f, std::min(1.0f, progress));
            const float cosVal = 0.5f * (1.0f + std::cos(3.14159265f * clampedProgress));
            currentLr_ = minLr_ + (peakLr_ - minLr_) * cosVal;
        }

        // Hard clamp — should never be needed but prevents sign-flip edge cases.
        currentLr_ = std::max(minLr_, std::min(peakLr_, currentLr_));
    }

    float currentLr() const noexcept { return currentLr_; }

    // Convenience: compute LR for any step without mutating state (for logging).
    float lrAt(int globalStep) const {
        LRScheduler tmp{ peakLr_, minLr_, maxSteps_, warmupSteps_, decayStartStep_, decaySteps_ };
        tmp.step(globalStep);
        return tmp.currentLr();
    }

private:
    float peakLr_ = 1e-4f;
    float minLr_ = 1e-6f;
    int   maxSteps_ = 500;
    int   warmupSteps_ = 50;
    int   decayStartStep_ = -1;
    int   decaySteps_ = -1;
    float currentLr_ = 0.0f;
};
