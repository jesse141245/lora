#include "backend_runtime_policy_builder.h"

#include <algorithm>
#include "env_utils.h"

namespace {

std::optional<BackendRuntimeProfileKind> backendRuntimeProfileOverride()
{
    const auto value = envutil::tryGetEnvString("QLORA_TEST_BACKEND_RUNTIME_PROFILE");
    if (!value) {
        return std::nullopt;
    }

    const std::string lowered = envutil::lowerCopy(*value);
    if (lowered == "legacy" || lowered == "legacy_compat") {
        return BackendRuntimeProfileKind::LegacyCompat;
    }
    if (lowered == "compact" || lowered == "short" || lowered == "short_seq_compact") {
        return BackendRuntimeProfileKind::ShortSeqCompact;
    }
    if (lowered == "long" || lowered == "long_seq_balanced_avx2") {
        return BackendRuntimeProfileKind::LongSeqBalancedAvx2;
    }
    return std::nullopt;
}

std::optional<BackendRuntimeProfileKind> backendLongSeqProfileOverride()
{
    const auto value = envutil::tryGetEnvString("QLORA_TEST_BACKEND_LONGSEQ_PROFILE");
    if (!value) {
        return std::nullopt;
    }

    const std::string lowered = envutil::lowerCopy(*value);
    if (lowered == "legacy" || lowered == "legacy_compat") {
        return BackendRuntimeProfileKind::LegacyCompat;
    }
    if (lowered == "compact" || lowered == "short" || lowered == "short_seq_compact") {
        return BackendRuntimeProfileKind::ShortSeqCompact;
    }
    if (lowered == "long" || lowered == "long_seq_balanced_avx2") {
        return BackendRuntimeProfileKind::LongSeqBalancedAvx2;
    }
    return std::nullopt;
}

BackendRuntimeContext resolveBackendRuntimeContext(
    std::size_t activeComputeT,
    std::size_t tTile)
{
    BackendRuntimeContext context{};
    context.activeComputeT = activeComputeT;
    context.tTile = tTile;
    if (const auto forcedProfile = backendRuntimeProfileOverride()) {
        context.profileKind = *forcedProfile;
        return context;
    }
    if (tTile >= 4 && activeComputeT >= 128) {
        context.profileKind =
            backendLongSeqProfileOverride().value_or(BackendRuntimeProfileKind::LongSeqBalancedAvx2);
        return context;
    }
    if (activeComputeT >= 128) {
        context.profileKind = BackendRuntimeProfileKind::ShortSeqCompact;
        return context;
    }
    context.profileKind = BackendRuntimeProfileKind::LegacyCompat;
    return context;
}

BackendRuntimePolicy buildDefaultBackendRuntimePolicy(
    const BackendRuntimeContext& context)
{
    BackendRuntimePolicy policy{};
    policy.context = context;
    auto& bitnet = policy.bitnetMatmul;
    auto& cpu = policy.cpuBackend;

    const std::size_t activeComputeT = context.activeComputeT;
    const std::size_t tTile = context.tTile;
    const bool longSeqBalanced = context.profileKind == BackendRuntimeProfileKind::LongSeqBalancedAvx2;
    const bool shortSeqCompact = context.profileKind == BackendRuntimeProfileKind::ShortSeqCompact;

    bitnet.fusedX4Forward = longSeqBalanced;
    bitnet.fusedX2SingleForward = false;
    bitnet.denseX2ForwardAccum = !longSeqBalanced && tTile <= 2 && activeComputeT >= 128;
    bitnet.parallelSingleForwardQuant = false;
    bitnet.parallelMultiForwardQuant = false;
    bitnet.fusedQkvForwardOverwrite = longSeqBalanced;
    bitnet.fusedQkvForwardTripleX4Accum = longSeqBalanced;

    bitnet.fusedX4Backward = longSeqBalanced;
    bitnet.fusedX4SingleBackward = longSeqBalanced;
    bitnet.fusedX4SingleBackwardOverwrite = true;
    bitnet.fusedX4SingleBackwardBlock32 = longSeqBalanced;
    bitnet.fusedX2SingleBackward = !longSeqBalanced && tTile <= 2 && activeComputeT >= 128;
    bitnet.fusedX2DualBackward = !longSeqBalanced && tTile <= 2 && activeComputeT >= 128;
    bitnet.parallelSingleBackwardQuant = longSeqBalanced;
    bitnet.parallelMultiBackwardQuant = false;
    bitnet.singleTeamBackward = !longSeqBalanced && tTile <= 2 && activeComputeT >= 128;
    bitnet.masterDualChunkQuant = true;

    cpu.activePrefixCompute = false;
    cpu.exactMaskedTileSkip = false;
    cpu.prefixMaskFastpath = false;
    cpu.rmsnormBackwardAccum = false;
    cpu.parallelAttentionPrep = activeComputeT >= 128;
    cpu.parallelAttentionLoraBackward = activeComputeT >= 128;
    cpu.ffnGemm = false;
    cpu.ffnBitnet = true;
    cpu.bitnetFfnStreamedPrefixBackward = false;
    cpu.bitnetFfnStreamedPrefixChunkT = 32;
    cpu.ffnBackwardFuseResidualAdd = activeComputeT >= 128;
    cpu.exactChunkedFfnBackward = false;
    cpu.lmHeadBackwardLossMaskSkip = false;

    if (longSeqBalanced) {
        cpu.exactFusedQkvForwardTileRows = std::max<std::size_t>(tTile, 16);
        cpu.exactWdownBackwardTileRows = std::max<std::size_t>(tTile, 16);
        cpu.exactLmHeadBackwardTileRows = std::max<std::size_t>(tTile, 256);
        cpu.exactFusedQkvBackwardTileRows = std::max<std::size_t>(tTile, 32);
        cpu.lmHeadBackwardHChunk = 64;
        cpu.exactChunkedFfnBackwardRows = std::max<std::size_t>(tTile, 32);
    }
    else if (shortSeqCompact) {
        cpu.exactFusedQkvForwardTileRows = tTile;
        cpu.exactWdownBackwardTileRows = std::max<std::size_t>(tTile, 16);
        cpu.exactLmHeadBackwardTileRows = std::max<std::size_t>(tTile, 256);
        cpu.exactFusedQkvBackwardTileRows = std::max<std::size_t>(tTile, 32);
        cpu.lmHeadBackwardHChunk = 64;
        cpu.exactChunkedFfnBackwardRows = std::max<std::size_t>(tTile, 32);
    }
    else {
        cpu.exactFusedQkvForwardTileRows = tTile;
        cpu.exactWdownBackwardTileRows = tTile;
        cpu.exactLmHeadBackwardTileRows = tTile;
        cpu.exactFusedQkvBackwardTileRows = tTile;
        cpu.lmHeadBackwardHChunk = 64;
        cpu.exactChunkedFfnBackwardRows = tTile;
    }

    return policy;
}

void applyBackendRuntimePolicyOverrides(BackendRuntimePolicy& policy)
{
    auto& bitnet = policy.bitnetMatmul;
    auto& cpu = policy.cpuBackend;

    bitnet.fusedX4Forward = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_X4_FORWARD").value_or(*bitnet.fusedX4Forward);
    bitnet.fusedX2SingleForward = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_X2_FORWARD_SINGLE").value_or(*bitnet.fusedX2SingleForward);
    bitnet.denseX2ForwardAccum = envutil::envBoolOverride("QLORA_TEST_BITNET_DENSE_X2_FORWARD_ACCUM").value_or(*bitnet.denseX2ForwardAccum);
    bitnet.parallelSingleForwardQuant = envutil::envBoolOverride("QLORA_TEST_BITNET_PARALLEL_FWD_QUANT_SINGLE").value_or(*bitnet.parallelSingleForwardQuant);
    bitnet.parallelMultiForwardQuant = envutil::envBoolOverride("QLORA_TEST_BITNET_PARALLEL_FWD_QUANT_MULTI").value_or(*bitnet.parallelMultiForwardQuant);
    bitnet.fusedQkvForwardOverwrite = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_OVERWRITE").value_or(*bitnet.fusedQkvForwardOverwrite);
    bitnet.fusedQkvForwardTripleX4Accum = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_X4_TRIPLE_ACCUM").value_or(*bitnet.fusedQkvForwardTripleX4Accum);

    bitnet.fusedX4Backward = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_X4_BACKWARD").value_or(*bitnet.fusedX4Backward);
    bitnet.fusedX4SingleBackward = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE").value_or(*bitnet.fusedX4SingleBackward);
    bitnet.fusedX4SingleBackwardOverwrite = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE_OVERWRITE").value_or(*bitnet.fusedX4SingleBackwardOverwrite);
    bitnet.fusedX4SingleBackwardBlock32 = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE_BLOCK32").value_or(*bitnet.fusedX4SingleBackwardBlock32);
    bitnet.fusedX2SingleBackward = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_X2_BACKWARD_SINGLE").value_or(*bitnet.fusedX2SingleBackward);
    bitnet.fusedX2DualBackward = envutil::envBoolOverride("QLORA_TEST_BITNET_FUSED_X2_BACKWARD_DUAL").value_or(*bitnet.fusedX2DualBackward);
    bitnet.parallelSingleBackwardQuant = envutil::envBoolOverride("QLORA_TEST_BITNET_PARALLEL_BWD_QUANT").value_or(*bitnet.parallelSingleBackwardQuant);
    bitnet.parallelMultiBackwardQuant = envutil::envBoolOverride("QLORA_TEST_BITNET_PARALLEL_BWD_QUANT_MULTI").value_or(*bitnet.parallelMultiBackwardQuant);
    bitnet.singleTeamBackward = envutil::envBoolOverride("QLORA_TEST_BITNET_SINGLE_TEAM_BWD_SINGLE").value_or(*bitnet.singleTeamBackward);
    bitnet.masterDualChunkQuant = envutil::envBoolOverride("QLORA_TEST_BITNET_DUAL_BWD_MASTER_QUANT").value_or(*bitnet.masterDualChunkQuant);
    bitnet.dualBackwardQuantChunkRows = envutil::envSizeTOverride("QLORA_TEST_BITNET_DUAL_BWD_QUANT_CHUNK_ROWS");
    bitnet.dualBackwardWorkTileRows = envutil::envSizeTOverride("QLORA_TEST_BITNET_DUAL_BWD_WORK_TILE_ROWS");

    cpu.activePrefixCompute = envutil::envBoolOverride("QLORA_TEST_ACTIVE_PREFIX_COMPUTE").value_or(*cpu.activePrefixCompute);
    cpu.exactMaskedTileSkip = envutil::envBoolOverride("QLORA_TEST_I2S_MASK_SKIP").value_or(*cpu.exactMaskedTileSkip);
    cpu.prefixMaskFastpath = envutil::envBoolOverride("QLORA_TEST_PREFIX_MASK_FASTPATH").value_or(*cpu.prefixMaskFastpath);
    cpu.rmsnormBackwardAccum = envutil::envBoolOverride("QLORA_TEST_RMSNORM_BWD_ACCUM").value_or(*cpu.rmsnormBackwardAccum);
    cpu.parallelAttentionPrep = envutil::envBoolOverride("QLORA_TEST_PARALLEL_ATTN_PREP").value_or(*cpu.parallelAttentionPrep);
    cpu.parallelAttentionLoraBackward = envutil::envBoolOverride("QLORA_TEST_PARALLEL_ATTN_LORA_BWD").value_or(*cpu.parallelAttentionLoraBackward);
    cpu.ffnGemm = envutil::envBoolOverride("QLORA_TEST_FFN_GEMM").value_or(*cpu.ffnGemm);
    cpu.ffnBitnet = envutil::envBoolOverride("QLORA_TEST_BITNET_FFN").value_or(*cpu.ffnBitnet);
    cpu.bitnetFfnStreamedPrefixBackward = envutil::envBoolOverride("QLORA_TEST_BITNET_FFN_BWD_STREAM_PREFIX").value_or(*cpu.bitnetFfnStreamedPrefixBackward);
    cpu.ffnBackwardFuseResidualAdd = envutil::envBoolOverride("QLORA_TEST_FFN_BWD_FUSE_RESIDUAL_ADD").value_or(*cpu.ffnBackwardFuseResidualAdd);
    cpu.exactChunkedFfnBackward = envutil::envBoolOverride("QLORA_TEST_BITNET_FFN_BWD_CHUNKED").value_or(*cpu.exactChunkedFfnBackward);
    cpu.lmHeadBackwardLossMaskSkip = envutil::envBoolOverride("QLORA_TEST_LM_HEAD_BWD_LOSS_MASK_SKIP").value_or(*cpu.lmHeadBackwardLossMaskSkip);
    cpu.lmHeadBackwardRowCompactOverride = envutil::envBoolOverride("QLORA_TEST_LM_HEAD_BWD_ROW_COMPACT");
    cpu.bitnetFfnStreamedPrefixChunkT = envutil::envSizeTOverride("QLORA_TEST_BITNET_FFN_BWD_STREAM_CHUNK_T").value_or(*cpu.bitnetFfnStreamedPrefixChunkT);
    cpu.exactFusedQkvForwardTileRows = envutil::envSizeTOverride("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_TILE_ROWS").value_or(*cpu.exactFusedQkvForwardTileRows);
    cpu.exactWdownBackwardTileRows = envutil::envSizeTOverride("QLORA_TEST_BITNET_WDOWN_BWD_TILE_ROWS").value_or(*cpu.exactWdownBackwardTileRows);
    cpu.exactLmHeadBackwardTileRows = envutil::envSizeTOverride("QLORA_TEST_BITNET_LM_HEAD_BWD_TILE_ROWS").value_or(*cpu.exactLmHeadBackwardTileRows);
    cpu.exactFusedQkvBackwardTileRows = envutil::envSizeTOverride("QLORA_TEST_BITNET_FUSED_QKV_BACKWARD_TILE_ROWS").value_or(*cpu.exactFusedQkvBackwardTileRows);
    cpu.lmHeadBackwardHChunk = envutil::envSizeTOverride("QLORA_TEST_LMHEAD_BWD_HCHUNK").value_or(*cpu.lmHeadBackwardHChunk);
    cpu.exactChunkedFfnBackwardRows = envutil::envSizeTOverride("QLORA_TEST_BITNET_FFN_BWD_CHUNK_ROWS").value_or(*cpu.exactChunkedFfnBackwardRows);
}

} // namespace

BackendRuntimePolicy buildBackendRuntimePolicy(
    std::size_t activeComputeT,
    std::size_t tTile)
{
    const BackendRuntimeContext context =
        resolveBackendRuntimeContext(activeComputeT, tTile);
    BackendRuntimePolicy policy =
        buildDefaultBackendRuntimePolicy(context);
    applyBackendRuntimePolicyOverrides(policy);
    return policy;
}
