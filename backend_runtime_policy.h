#pragma once

#include <cstddef>
#include <optional>

enum class BackendRuntimeProfileKind {
    LegacyCompat,
    ShortSeqCompact,
    LongSeqBalancedAvx2,
};

struct BackendRuntimeContext {
    BackendRuntimeProfileKind profileKind = BackendRuntimeProfileKind::LegacyCompat;
    std::size_t activeComputeT = 0;
    std::size_t tTile = 0;
};

inline const char* backendRuntimeProfileKindName(BackendRuntimeProfileKind kind) {
    switch (kind) {
    case BackendRuntimeProfileKind::LegacyCompat:
        return "legacy_compat";
    case BackendRuntimeProfileKind::ShortSeqCompact:
        return "short_seq_compact";
    case BackendRuntimeProfileKind::LongSeqBalancedAvx2:
        return "long_seq_balanced_avx2";
    }
    return "unknown";
}

struct BitNetMatmulRuntimePolicy {
    std::optional<bool> fusedX4Forward;
    std::optional<bool> fusedX2SingleForward;
    std::optional<bool> denseX2ForwardAccum;
    std::optional<bool> parallelSingleForwardQuant;
    std::optional<bool> parallelMultiForwardQuant;
    std::optional<bool> fusedQkvForwardOverwrite;
    std::optional<bool> fusedQkvForwardTripleX4Accum;

    std::optional<bool> fusedX4Backward;
    std::optional<bool> fusedX4SingleBackward;
    std::optional<bool> fusedX4SingleBackwardOverwrite;
    std::optional<bool> fusedX4SingleBackwardBlock32;
    std::optional<bool> fusedX2SingleBackward;
    std::optional<bool> fusedX2DualBackward;
    std::optional<bool> parallelSingleBackwardQuant;
    std::optional<bool> parallelMultiBackwardQuant;
    std::optional<bool> singleTeamBackward;
    std::optional<bool> masterDualChunkQuant;

    std::optional<std::size_t> dualBackwardQuantChunkRows;
    std::optional<std::size_t> dualBackwardWorkTileRows;
};

struct CpuBackendRuntimePolicy {
    std::optional<bool> activePrefixCompute;
    std::optional<bool> exactMaskedTileSkip;
    std::optional<bool> prefixMaskFastpath;
    std::optional<bool> rmsnormBackwardAccum;
    std::optional<bool> parallelAttentionPrep;
    std::optional<bool> parallelAttentionLoraBackward;
    std::optional<bool> ffnGemm;
    std::optional<bool> ffnBitnet;
    std::optional<bool> bitnetFfnStreamedPrefixBackward;
    std::optional<bool> ffnBackwardFuseResidualAdd;
    std::optional<bool> exactChunkedFfnBackward;
    std::optional<bool> lmHeadBackwardLossMaskSkip;
    std::optional<bool> lmHeadBackwardRowCompactOverride;

    std::optional<std::size_t> bitnetFfnStreamedPrefixChunkT;
    std::optional<std::size_t> exactFusedQkvForwardTileRows;
    std::optional<std::size_t> exactWdownBackwardTileRows;
    std::optional<std::size_t> exactLmHeadBackwardTileRows;
    std::optional<std::size_t> exactFusedQkvBackwardTileRows;
    std::optional<std::size_t> lmHeadBackwardHChunk;
    std::optional<std::size_t> exactChunkedFfnBackwardRows;
};

struct BackendRuntimePolicy {
    BackendRuntimeContext context;
    BitNetMatmulRuntimePolicy bitnetMatmul;
    CpuBackendRuntimePolicy cpuBackend;
};

inline thread_local const BackendRuntimePolicy* gCurrentBackendRuntimePolicy = nullptr;

inline const BackendRuntimePolicy* currentBackendRuntimePolicy() {
    return gCurrentBackendRuntimePolicy;
}

class ScopedBackendRuntimePolicy {
public:
    explicit ScopedBackendRuntimePolicy(const BackendRuntimePolicy& policy)
        : previous_(gCurrentBackendRuntimePolicy) {
        gCurrentBackendRuntimePolicy = &policy;
    }

    ~ScopedBackendRuntimePolicy() {
        gCurrentBackendRuntimePolicy = previous_;
    }

    ScopedBackendRuntimePolicy(const ScopedBackendRuntimePolicy&) = delete;
    ScopedBackendRuntimePolicy& operator=(const ScopedBackendRuntimePolicy&) = delete;

private:
    const BackendRuntimePolicy* previous_ = nullptr;
};
