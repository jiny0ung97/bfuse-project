
#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include <map>

#include "fuse/Contexts.h"
#include "fuse/Algorithms.h"

using namespace std;
//---------------------------------------------------------------------------
namespace fuse {
//---------------------------------------------------------------------------
namespace contexts {
//---------------------------------------------------------------------------
void KernelInfo::print(const string &KName) const
{
  cout << "[KernelInfo]\n";
  cout << KName << ":\n";
  cout << "  KernelName: " << KernelName_ << "\n";
  cout << "  Barriers: " << HasBarriers_ << "\n";
  cout << "  GridDim:\n";
  cout << "    X: " << GridDim_.X << "\n";
  cout << "    Y: " << GridDim_.Y << "\n";
  cout << "    Z: " << GridDim_.Z << "\n";
  cout << "  BlockDim:\n";
  cout << "    X: " << BlockDim_.X << "\n";
  cout << "    Y: " << BlockDim_.Y << "\n";
  cout << "    Z: " << BlockDim_.Z << "\n";
  cout << "  Reg: " << Reg_ << "\n";
  cout << "  ExecTime: " << ExecTime_ << "\n";
}
//---------------------------------------------------------------------------
void FusionInfo::print() const
{
  cout << "[FusionInfo]\n";
  cout << "- File: " << File_ << "\n";
  cout << "  Kernels:\n";
  for (auto& KName : Kernels_) {
    cout << "   - " << KName << "\n";
  }
}
//---------------------------------------------------------------------------
void FusionContext::print() const
{
  cout << "================= FusionContext =================\n";
  for (auto& KName : Kernels_) {
    auto& KernelInfo    = KernelInfoMap_.at(KName);

    cout << "/// " << KName << " ///\n\n";
    KernelInfo.print(KName);
  }
}
//---------------------------------------------------------------------------
FusionContext FusionContext::create(FusionInfo &FInfo, map<string, KernelInfo> &KInfoMap, bool bfuse)
{
  int                     TotalSM;
  vector<string>          Kernels;
  map<string, KernelInfo> KernelInfoMap;
  string                  FusedKernelName;
  FusionContext           FContext;

  // Temp: V100
  TotalSM = 84;

  FusedKernelName = "";
  for (auto& KName : FInfo.Kernels_) {
    auto& Info = KInfoMap.at(KName);

    Kernels.push_back(KName);
    KernelInfoMap[KName] = Info;
    FusedKernelName      += KName + "_";
  }
  if (bfuse) {
    FusedKernelName += "fused_bfuse";
    // auto [FusedBlockDeclStr, FusedCondStrMap, FusedGridDim, FusedBlockDim] = algorithms::coarseInterleavePattern(FInfo.Kernels_, KInfoMap, TotalSM);
    auto [FusedBlockDeclStr, FusedCondStrMap, FusedGridDim, FusedBlockDim] = algorithms::coarseInterleaveWithoutSMPattern(FInfo.Kernels_, KInfoMap);

    FContext = FusionContext{TotalSM, move(Kernels), move(KernelInfoMap), move(FusedKernelName),
                             move(FusedGridDim), move(FusedBlockDim), move(FusedBlockDeclStr), move(FusedCondStrMap)};
  } else {
    FusedKernelName += "fused_hfuse";
    auto [FusedBlockDeclStrMap, FusedCondStrMap, FusedGridDim, FusedBlockDim] = algorithms::hfusePattern(FInfo.Kernels_, KInfoMap);

    FContext = FusionContext{TotalSM, move(Kernels), move(KernelInfoMap), move(FusedKernelName),
                             move(FusedGridDim), move(FusedBlockDim), move(FusedBlockDeclStrMap), move(FusedCondStrMap)};
  }

  return FContext;
}
//---------------------------------------------------------------------------
} // namespace contexts
} // namespace fuse
//---------------------------------------------------------------------------