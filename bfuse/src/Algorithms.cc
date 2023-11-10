
#include <algorithm>
#include <tuple>
#include <map>

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Algorithms.h"

using namespace std;

using namespace bfuse::contexts;
//---------------------------------------------------------------------------
namespace bfuse {
namespace algorithms {
//---------------------------------------------------------------------------
map<string, KernelContext> zigZagBlockPattern(vector<string> &Kernels, map<string, KernelInfo> &KInfoMap)
{
  map<string, int>           CurBounds;
  map<string, int>           EndBounds;
  map<string, KernelContext> KernelContextMap;

  // Intialize containers
  for (auto& KName : Kernels) {
    auto& Info = KInfoMap.at(KName);

    CurBounds[KName] = 0;
    EndBounds[KName] = Info.gridDim.size();  

    IdxBoundPairTy ThreadIdxInfo;
    
    ThreadIdxInfo           = make_pair(0, Info.blockDim.size());
    KernelContextMap[KName] = KernelContext{move(ThreadIdxInfo)};
  }

  /*
   * <Total SM Counts>
   * V100 : 84
   * 
   */
  int TotalSM     = 84;
  int Idx         = 0;
  int TotalBounds = 0;
  bool LastLoop   = false;

  // Calcalate blockIdx boundary and other blocks
  while(true) {
    auto& KName        = Kernels[Idx];
    auto& Context      = KernelContextMap.at(KName);
    auto& BlockIdxInfo = Context.blockIdxInfo;
    auto& OtherBlocks  = Context.otherBlocks;

    int Stride = EndBounds[KName] - CurBounds[KName];

    if (!LastLoop && Stride > TotalSM)
      Stride = TotalSM;
    
    BlockIdxInfo.emplace_back(TotalBounds, TotalBounds + Stride);
    OtherBlocks.push_back(TotalBounds - CurBounds[KName]);

    if (LastLoop)
      break;

    CurBounds[KName] += Stride;
    TotalBounds      += Stride;

    if (CurBounds[KName] == EndBounds[KName])
      LastLoop = true;

    Idx = (Idx + 1) % Kernels.size();
  }

  return KernelContextMap;
}
//---------------------------------------------------------------------------
map<string, KernelContext> sequentialBlockPattern(vector<string> &Kernels, map<string, KernelInfo> &KInfoMap)
{
  map<string, KernelContext> KernelContextMap;
  int TotalBounds = 0;

  // Intialize containers & Calcalate blockIdx boundary
  for (auto& KName : Kernels) {
    auto& Info = KInfoMap.at(KName);

    IdxBoundPairTy         ThreadIdxInfo;
    vector<IdxBoundPairTy> BlockIdxInfo;
    vector<int>            OtherBlocks;

    ThreadIdxInfo = make_pair(0, Info.blockDim.size());
    BlockIdxInfo.push_back(make_pair(TotalBounds, TotalBounds + Info.gridDim.size()));
    OtherBlocks.push_back(TotalBounds);

    KernelContextMap[KName] = KernelContext{move(ThreadIdxInfo), move(BlockIdxInfo), move(OtherBlocks)};

    TotalBounds += Info.gridDim.size();
  }

  return KernelContextMap;
}
//---------------------------------------------------------------------------
tuple<VarListTy, VarListTy, USRsListTy> getNewParmLists(const AnalysisContext &AContext)
{
  VarListTy NewParams;
  VarListTy PrevParams;
  USRsListTy USRs;

  for (auto &KName : AContext.Kernels) {
    if (AContext.ParmListMap.find(KName) == AContext.ParmListMap.end()) {
      continue;
    }
    
    auto &PrevParmList = AContext.ParmListMap.at(KName);
    auto &USRsList     = AContext.ParmUSRsListMap.at(KName);

    VarListTy NewParmList{PrevParmList.size()};
    transform(PrevParmList.begin(), PrevParmList.end(),
              NewParmList.begin(),
              [&KName](const string &PName) {
                return KName + "_" + PName + "_";
              });

    NewParams.insert(NewParams.end(),
                     NewParmList.begin(), NewParmList.end());
    PrevParams.insert(PrevParams.end(),
                      PrevParmList.begin(), PrevParmList.end());
    USRs.insert(USRs.end(),
                USRsList.begin(), USRsList.end());
  }
  return make_tuple(NewParams, PrevParams, USRs);
}
//---------------------------------------------------------------------------
tuple<VarListMapTy, USRsListMapTy, SizeListMapTy, std::string>
getShrdVarAnalysis(const AnalysisContext &AContext, VarListMapTy &VarListMap, USRsListMapTy &USRsListMap, SizeListMapTy &SizeListMap)
{
  // Sorting function
  auto SortFunc = [](auto &Container, auto &Criteria) {
    return [&Container, &Criteria](auto &A, auto &B) {
      int AIdx = find(Container.begin(), Container.end(), A) - Container.begin();
      int BIdx = find(Container.begin(), Container.end(), B) - Container.begin();
      return Criteria[AIdx] > Criteria[BIdx];
    };
  };

  // Copy & Sort the containers
  VarListMapTy ShrdVarListMap      = VarListMap;
  USRsListMapTy ShrdVarUSRsListMap = USRsListMap;
  SizeListMapTy ShrdVarSizeListMap = SizeListMap;

  for (auto &KName : AContext.Kernels) {
    // If kernel has no shared memory variable, skip it.
    if (ShrdVarListMap.find(KName) == ShrdVarListMap.end())
      continue;

    auto &ShrdVarList     = ShrdVarListMap.at(KName);
    auto &ShrdVarUSRsList = ShrdVarUSRsListMap.at(KName);
    auto &ShrdVarSizeList = ShrdVarSizeListMap.at(KName);

    auto &OldShrdVarList     = VarListMap.at(KName);
    auto &OldShrdVarUSRsList = USRsListMap.at(KName);
    auto &OldShrdVarSizeList = SizeListMap.at(KName);

    sort(ShrdVarList.begin(), ShrdVarList.end(),
         SortFunc(OldShrdVarList, OldShrdVarSizeList));
    sort(ShrdVarUSRsList.begin(), ShrdVarUSRsList.end(),
         SortFunc(OldShrdVarUSRsList, OldShrdVarSizeList));
    sort(ShrdVarSizeList.begin(), ShrdVarSizeList.end(),
         SortFunc(OldShrdVarSizeList, OldShrdVarSizeList));
  }

  // Generate new shared memory declarations for fused kernel
  string ShrdDeclStr;
  llvm::raw_string_ostream ShrdDeclStream{ShrdDeclStr};

  string VNameBase = "union_shared_";
  bool AllVisited;
  uint64_t MaxSize;

  for (long unsigned I = 0;; ++I) {
    AllVisited = true;
    MaxSize    = 0;

    for (auto &KName : AContext.Kernels) {
      auto &ShrdVarSizeList = ShrdVarSizeListMap.at(KName);

      if (I >= ShrdVarSizeList.size())
        continue;

      AllVisited = false;
      MaxSize    = ShrdVarSizeList[I] > MaxSize ? ShrdVarSizeList[I] : MaxSize;
    }

    if (AllVisited)
      break;

    // FIXME: need to print by more general methology
    // FORMAT: static float pad_temp_shared[2320] __attribute__((shared));
    ShrdDeclStream << "  static float union_shared_" << I << "_[" << MaxSize << "] __attribute__((shared));\n";
  }
  ShrdDeclStream << "\n";
  ShrdDeclStream.flush();

  return make_tuple(ShrdVarListMap, ShrdVarUSRsListMap, ShrdVarSizeListMap, ShrdDeclStr);
}
//---------------------------------------------------------------------------
tuple<VarListTy, VarListTy, USRsListTy> getNewShrdVarLists(const AnalysisContext &AContext)
{
  VarListTy  NewShrdVars;
  VarListTy  PrevShrdVars;
  USRsListTy USRs;

  string NewNameBase = "union_shared_";

  for (auto &KName : AContext.Kernels) {
    // If kernel has no shared memory variable, skip it.
    if (AContext.ShrdVarListMap.find(KName) == AContext.ShrdVarListMap.end())
      continue;

    auto &PrevShrdVarList = AContext.ShrdVarListMap.at(KName);
    auto &USRsList        = AContext.ShrdVarUSRsListMap.at(KName);

    VarListTy NewShrdVarList;
    for (long unsigned I = 0; I < PrevShrdVarList.size(); ++I) {
      NewShrdVarList.push_back(NewNameBase + to_string(I) + "_");
    }

    NewShrdVars.insert(NewShrdVars.end(),
                       NewShrdVarList.begin(), NewShrdVarList.end());
    PrevShrdVars.insert(PrevShrdVars.end(),
                        PrevShrdVarList.begin(), PrevShrdVarList.end());
    USRs.insert(USRs.end(),
                USRsList.begin(), USRsList.end());
  }
  return make_tuple(NewShrdVars, PrevShrdVars, USRs);
}
//---------------------------------------------------------------------------
} // namespace algorithms
} // namespace bfuse
//---------------------------------------------------------------------------