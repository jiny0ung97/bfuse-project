
#pragma once

#include <string>
#include <map>
#include <tuple>

#include "fuse/Contexts.h"
//---------------------------------------------------------------------------
namespace fuse {
namespace algorithms {
//---------------------------------------------------------------------------
std::tuple<std::map<std::string, std::string>, std::map<std::string, std::string>, contexts::GridDim, contexts::BlockDim>
hfusePattern(std::vector<std::string> &Kernels, std::map<std::string, contexts::KernelInfo> &KernelInfoMap);
//---------------------------------------------------------------------------
std::tuple<std::string, std::map<std::string, std::string>, contexts::GridDim, contexts::BlockDim>
coarseInterleavePattern(std::vector<std::string> &Kernels, std::map<std::string, contexts::KernelInfo> &KernelInfoMap, int TotalSM);
//---------------------------------------------------------------------------
std::tuple<std::string, std::map<std::string, std::string>, contexts::GridDim, contexts::BlockDim>
coarseInterleaveWithoutSMPattern(std::vector<std::string> &Kernels, std::map<std::string, contexts::KernelInfo> &KernelInfoMap);
//---------------------------------------------------------------------------
} // namespace algorithms
} // namespace fuse
//---------------------------------------------------------------------------