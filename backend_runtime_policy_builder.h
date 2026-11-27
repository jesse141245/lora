#pragma once

#include <cstddef>

#include "backend_runtime_policy.h"

BackendRuntimePolicy buildBackendRuntimePolicy(
    std::size_t activeComputeT,
    std::size_t tTile);
