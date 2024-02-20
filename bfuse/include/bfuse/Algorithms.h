
#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <map>

#include "bfuse/Contexts.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace algorithms {
//---------------------------------------------------------------------------
using VarListTy     = contexts::VarListTy;
using USRsListTy    = contexts::USRsListTy;
using SizeListTy    = contexts::SizeListTy;
using VarListMapTy  = contexts::VarListMapTy;
using USRsListMapTy = contexts::USRsListMapTy;
using SizeListMapTy = contexts::SizeListMapTy;
//---------------------------------------------------------------------------
std::map<std::string, contexts::KernelContext> zigZagBlockPattern(std::vector<std::string> &Kernels,
                                                                  std::map<std::string, contexts::KernelInfo> &KInfoMap);
//---------------------------------------------------------------------------
std::map<std::string, contexts::KernelContext> sequentialBlockPattern(std::vector<std::string> &Kernels,
                                                                      std::map<std::string, contexts::KernelInfo> &KInfoMap);
//---------------------------------------------------------------------------
std::string oldInterleaveBlockPattern(contexts::FusionContext &FContext);
//---------------------------------------------------------------------------
std::string newInterleaveBlockPattern(contexts::FusionContext &FContext);
//---------------------------------------------------------------------------
std::tuple<VarListTy, VarListTy, USRsListTy> getNewParmLists(const contexts::AnalysisContext &AContext);
//---------------------------------------------------------------------------
std::tuple<VarListMapTy, USRsListMapTy, SizeListMapTy, std::string>
getShrdVarAnalysis(const contexts::AnalysisContext &AContext, VarListMapTy &VarListMap,
                   USRsListMapTy &USRsListMap, SizeListMapTy &SizeListMap);
//---------------------------------------------------------------------------
std::tuple<VarListTy, VarListTy, USRsListTy> getNewShrdVarLists(const contexts::AnalysisContext &AContext);
//---------------------------------------------------------------------------
} // namespace algorithms
} // namespace bfuse
//---------------------------------------------------------------------------