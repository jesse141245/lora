#include "backend_runtime_policy_json.h"
#include "env_utils.h"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_set>

namespace {

nlohmann::json optionalBool(const std::optional<bool>& value) {
    return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

nlohmann::json optionalSizeT(const std::optional<std::size_t>& value) {
    return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

std::optional<std::filesystem::path> runtimePolicySnapshotPath() {
    static const std::optional<std::filesystem::path> cached = []() -> std::optional<std::filesystem::path> {
        const auto value = envutil::tryGetEnvString("QLORA_RUNTIME_POLICY_SNAPSHOT_PATH");
        if (!value || value->empty()) {
            return std::nullopt;
        }
        return std::filesystem::path(*value);
    }();
    return cached;
}

} // namespace

namespace {

nlohmann::json backendRuntimePolicyToJson(const BackendRuntimePolicy& policy) {
    nlohmann::json j;
    j["context"] = {
        { "profileKind", backendRuntimeProfileKindName(policy.context.profileKind) },
        { "activeComputeT", policy.context.activeComputeT },
        { "tTile", policy.context.tTile },
    };

    j["bitnetMatmul"] = {
        { "fusedX4Forward", optionalBool(policy.bitnetMatmul.fusedX4Forward) },
        { "fusedX2SingleForward", optionalBool(policy.bitnetMatmul.fusedX2SingleForward) },
        { "denseX2ForwardAccum", optionalBool(policy.bitnetMatmul.denseX2ForwardAccum) },
        { "parallelSingleForwardQuant", optionalBool(policy.bitnetMatmul.parallelSingleForwardQuant) },
        { "parallelMultiForwardQuant", optionalBool(policy.bitnetMatmul.parallelMultiForwardQuant) },
        { "fusedQkvForwardOverwrite", optionalBool(policy.bitnetMatmul.fusedQkvForwardOverwrite) },
        { "fusedQkvForwardTripleX4Accum", optionalBool(policy.bitnetMatmul.fusedQkvForwardTripleX4Accum) },
        { "fusedX4Backward", optionalBool(policy.bitnetMatmul.fusedX4Backward) },
        { "fusedX4SingleBackward", optionalBool(policy.bitnetMatmul.fusedX4SingleBackward) },
        { "fusedX4SingleBackwardOverwrite", optionalBool(policy.bitnetMatmul.fusedX4SingleBackwardOverwrite) },
        { "fusedX4SingleBackwardBlock32", optionalBool(policy.bitnetMatmul.fusedX4SingleBackwardBlock32) },
        { "fusedX2SingleBackward", optionalBool(policy.bitnetMatmul.fusedX2SingleBackward) },
        { "fusedX2DualBackward", optionalBool(policy.bitnetMatmul.fusedX2DualBackward) },
        { "parallelSingleBackwardQuant", optionalBool(policy.bitnetMatmul.parallelSingleBackwardQuant) },
        { "parallelMultiBackwardQuant", optionalBool(policy.bitnetMatmul.parallelMultiBackwardQuant) },
        { "singleTeamBackward", optionalBool(policy.bitnetMatmul.singleTeamBackward) },
        { "masterDualChunkQuant", optionalBool(policy.bitnetMatmul.masterDualChunkQuant) },
        { "dualBackwardQuantChunkRows", optionalSizeT(policy.bitnetMatmul.dualBackwardQuantChunkRows) },
        { "dualBackwardWorkTileRows", optionalSizeT(policy.bitnetMatmul.dualBackwardWorkTileRows) },
    };

    j["cpuBackend"] = {
        { "activePrefixCompute", optionalBool(policy.cpuBackend.activePrefixCompute) },
        { "exactMaskedTileSkip", optionalBool(policy.cpuBackend.exactMaskedTileSkip) },
        { "prefixMaskFastpath", optionalBool(policy.cpuBackend.prefixMaskFastpath) },
        { "rmsnormBackwardAccum", optionalBool(policy.cpuBackend.rmsnormBackwardAccum) },
        { "parallelAttentionPrep", optionalBool(policy.cpuBackend.parallelAttentionPrep) },
        { "parallelAttentionLoraBackward", optionalBool(policy.cpuBackend.parallelAttentionLoraBackward) },
        { "ffnGemm", optionalBool(policy.cpuBackend.ffnGemm) },
        { "ffnBitnet", optionalBool(policy.cpuBackend.ffnBitnet) },
        { "bitnetFfnStreamedPrefixBackward", optionalBool(policy.cpuBackend.bitnetFfnStreamedPrefixBackward) },
        { "ffnBackwardFuseResidualAdd", optionalBool(policy.cpuBackend.ffnBackwardFuseResidualAdd) },
        { "exactChunkedFfnBackward", optionalBool(policy.cpuBackend.exactChunkedFfnBackward) },
        { "lmHeadBackwardLossMaskSkip", optionalBool(policy.cpuBackend.lmHeadBackwardLossMaskSkip) },
        { "lmHeadBackwardRowCompactOverride", optionalBool(policy.cpuBackend.lmHeadBackwardRowCompactOverride) },
        { "bitnetFfnStreamedPrefixChunkT", optionalSizeT(policy.cpuBackend.bitnetFfnStreamedPrefixChunkT) },
        { "exactFusedQkvForwardTileRows", optionalSizeT(policy.cpuBackend.exactFusedQkvForwardTileRows) },
        { "exactWdownBackwardTileRows", optionalSizeT(policy.cpuBackend.exactWdownBackwardTileRows) },
        { "exactLmHeadBackwardTileRows", optionalSizeT(policy.cpuBackend.exactLmHeadBackwardTileRows) },
        { "exactFusedQkvBackwardTileRows", optionalSizeT(policy.cpuBackend.exactFusedQkvBackwardTileRows) },
        { "lmHeadBackwardHChunk", optionalSizeT(policy.cpuBackend.lmHeadBackwardHChunk) },
        { "exactChunkedFfnBackwardRows", optionalSizeT(policy.cpuBackend.exactChunkedFfnBackwardRows) },
    };

    return j;
}

} // namespace

void recordBackendRuntimePolicySnapshot(const BackendRuntimePolicy& policy) {
    const auto snapshotPath = runtimePolicySnapshotPath();
    if (!snapshotPath) {
        return;
    }

    const nlohmann::json policyJson = backendRuntimePolicyToJson(policy);
    const std::string fingerprint = policyJson.dump();

    static std::mutex mutex;
    static std::unordered_set<std::string> seenPolicies;
    static nlohmann::json snapshotRoot = {
        { "schemaVersion", 1 },
        { "snapshotKind", "resolved_backend_runtime_policy_set" },
        { "policies", nlohmann::json::array() },
    };

    std::lock_guard<std::mutex> lock(mutex);
    if (!seenPolicies.insert(fingerprint).second) {
        return;
    }

    snapshotRoot["policies"].push_back(policyJson);

    std::filesystem::create_directories(snapshotPath->parent_path());
    std::ofstream out(*snapshotPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }
    out << snapshotRoot.dump(2);
}
