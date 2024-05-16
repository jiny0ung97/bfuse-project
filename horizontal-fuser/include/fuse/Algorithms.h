
#pragma once

#include <string>
#include <map>
#include <tuple>

#include "fuse/Contexts.h"
//---------------------------------------------------------------------------
namespace fuse {
namespace algorithms {
//---------------------------------------------------------------------------
std::tuple<std::string, std::map<std::string, std::string>, contexts::GridDim, contexts::BlockDim>
fineInterleavePattern(std::vector<std::string> &Kernels, std::map<std::string, contexts::KernelInfo> &KernelInfoMap, int TotalSM);
//---------------------------------------------------------------------------
} // namespace algorithms
} // namespace fuse
//---------------------------------------------------------------------------