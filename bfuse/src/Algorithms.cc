
#include <algorithm>
#include <tuple>

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Algorithms.h"

using namespace std;

using namespace bfuse::contexts;
//---------------------------------------------------------------------------
namespace bfuse {
namespace algorithms {
//---------------------------------------------------------------------------
tuple<VarListTy, VarListTy, USRsListTy> getNewParmLists(const AnalysisContext &AContext)
{
  VarListTy NewParams;
  VarListTy PrevParams;
  USRsListTy USRs;

  for (auto &KName : AContext.Kernels) {
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