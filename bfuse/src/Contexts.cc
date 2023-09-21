
#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include <map>

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"

using namespace std;
//---------------------------------------------------------------------------
namespace bfuse {
namespace contexts {
//---------------------------------------------------------------------------
void KernelInfo::print(const string& KName) const
{
  cout << "[KernelInfo]\n";
  cout << KName << "\n";
  cout << "  File: " << filePath << "\n";
  cout << "  Barriers: " << hasBarriers << "\n";
  cout << "  GridDim:\n";
  cout << "    X: " << gridDim.x << "\n";
  cout << "    Y: " << gridDim.y << "\n";
  cout << "    Z: " << gridDim.z << "\n";
  cout << "  BlockDim:\n";
  cout << "    X: " << blockDim.x << "\n";
  cout << "    Y: " << blockDim.y << "\n";
  cout << "    Z: " << blockDim.z << "\n\n";
}
//---------------------------------------------------------------------------
void FusionInfo::print() const
{
  cout << "[FusionInfo]\n";
  cout << "  - Kernels:\n";
  for (auto& KName : kernels) {
    cout << "    - " << KName << "\n";
  }
  cout << "\n";
}
//---------------------------------------------------------------------------
void KernelContext::print() const
{
  cout << "[KernelContext Info]\n";
  cout << "- ThreadIdxInfo: [" << threadIdxInfo.first << " ~ " << threadIdxInfo.second << ")\n";
  cout << "- BlockIdxInfo : ";
  for (auto& info : blockIdxInfo) {
    cout << "[" << info.first << " ~ " << info.second << ") ";
  }
  cout << "\n";
  cout << "- OtherBlocks  : ";
  for (auto n : otherBlocks) {
    cout << n << " ";
  }
  cout << "\n\n";
}
//---------------------------------------------------------------------------
void FusionContext::print() const
{
  cout << "================= FusionContext =================\n";
  for (auto& KName : kernels) {
    auto& KernelInfo    = kernelInfoMap.at(KName);
    auto& KernelContext = kernelContextMap.at(KName);

    cout << "/// " << KName << " ///\n\n";
    KernelInfo.print(KName);
    KernelContext.print();
  }
}
//---------------------------------------------------------------------------
void AnalysisContext::print() const
{
  // TODO : implement print() function

  // cout << "================= AnalysisContext =================\n";
  // for (auto& KName : kernels) {
  //   auto& PL = ParamListMap.at(KName);
  //   auto& UL = USRsListMap.at(KName);

  //   cout << "/// " << KName << " ///\n\n";
  //   cout << "Parameters: ";
  //   for (auto& P : PL) {
  //     cout << P << " ";
  //   }
  //   cout << "\n";

  //   cout << "USRs:\n";
  //   for (auto& UV : UL) {
  //     cout << "      ";
  //     for (auto& U : UV) {
  //       cout << U << " ";
  //     }
  //     cout << "\n";
  //   }
  //   cout << "\n";
  // }
}
//---------------------------------------------------------------------------
FusionContext FusionContext::create(FusionInfo &FInfo, map<string, KernelInfo> &KInfoMap)
{
  vector<string>             Kernels;
  map<string, KernelInfo>    KernelInfoMap;
  map<string, KernelContext> KernelContextMap;

  map<string, int> CurBounds;
  map<string, int> EndBounds;

  /*
   * <Total SM Counts>
   * V100 : 84
   * 
   */
  const int TotalSM = 84;

  int Idx         = 0;
  int TotalBounds = 0;
  bool LastLoop   = false;

  // Intialize containers
  for (auto& KName : FInfo.kernels) {
    auto& Info = KInfoMap.at(KName);

    CurBounds[KName]     = 0;
    EndBounds[KName]     = Info.gridDim.size();

    Kernels.push_back(KName);
    KernelInfoMap[KName]    = Info;
    KernelContextMap[KName] = KernelContext{make_pair(0, Info.blockDim.size())};
  }

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

  return FusionContext{move(Kernels), move(KernelInfoMap), move(KernelContextMap)};
}
//---------------------------------------------------------------------------
AnalysisContext AnalysisContext::create(FusionContext &FContext)
{
  // Lambda function
  auto PrintInfoToCondFunc = [](string V, auto &Info) {
    string Str;
    llvm::raw_string_ostream RawStream{Str};
    RawStream << "(" << V << " >= " << Info.first << " && "
              << V << " < " << Info.second << ")";
    RawStream.flush();
    return Str;
  };

  // Declarations
  vector<string>      Kernels;
  map<string, int>    ThreadNumMap;
  map<string, string> BranchConditionMap;

  string TmpBlockInfoString;
  string NewBlockInfoString;

  string NewFuncName = "";
  int MaxThreadBound = 0;

  // 1. Initialize Kernels & ThreadNumMap & NewFuncName & MaxThreadBound
  // -----------------------------------------------------------------
  // TODO: Add comments

  Kernels = FContext.kernels;
  for (auto &KName : FContext.kernels) {
    auto &KContext = FContext.kernelContextMap.at(KName);
    auto &TInfo    = KContext.threadIdxInfo;

    ThreadNumMap[KName] = TInfo.second;
    NewFuncName += KName + "_";
    MaxThreadBound = TInfo.second > MaxThreadBound ? TInfo.second : MaxThreadBound;
  }
  NewFuncName += "fused_";

  // 2. Initialize BranchConditionMap
  // -----------------------------------------------------------------
  // TODO: Add comments

  for (long unsigned I = 0; I < FContext.kernels.size(); ++I) {
    auto &KName    = FContext.kernels[I];
    auto &KContext = FContext.kernelContextMap.at(KName);
    auto &TInfo    = KContext.threadIdxInfo;

    string CondStr;
    llvm::raw_string_ostream CondStream{CondStr};

    // "KernelID_" condition check
    CondStream << "(KernelID_ == " << I << ") && ";

    // threadIdx condition check
    CondStream << "(" << PrintInfoToCondFunc("threadIdx.x", TInfo) << ")";
    CondStream.flush();

    BranchConditionMap[KName] = CondStr;
  }

  // 3. Initialize TmpBlockInfoString
  // -----------------------------------------------------------------
  // TODO: Add comments

  llvm::raw_string_ostream TmpVarStream{TmpBlockInfoString};
  TmpVarStream << "\n"
               << "  // FIXME: need to be deleted later\n"
               << "  int gridDim_x_  = 0; // temp declaration\n"
               << "  int blockIdx_x_ = 0; // temp declaration\n";
  TmpVarStream.flush();

  // 4. Initialize NewBlockInfoString
  // -----------------------------------------------------------------
  // TODO: Add comments

  llvm::raw_string_ostream VarStream{NewBlockInfoString};

  // Comments
  VarStream << "  /*\n"
            << "   * KernelID_ means...\n";

  for (long unsigned I = 0; I < FContext.kernels.size(); ++I) {
    string &KName = FContext.kernels[I];
    VarStream << "   * " << I << ": " << KName << "\n";
  }
  VarStream << "   */\n";

  // Declarations
  VarStream << "  int gridDim_x_;\n"
            << "  int blockIdx_x_;\n"
            << "  int Others_;\n"
            << "  int KernelID_;\n"
            << "  \n";

  bool IsAllVisited = false;
  for (long unsigned VI = 0; !IsAllVisited; ++VI) {
    IsAllVisited = true;

    for (long unsigned KI = 0; KI < FContext.kernels.size(); ++KI) {
      auto &KName    = FContext.kernels[KI];
      auto &KInfo    = FContext.kernelInfoMap.at(KName);
      auto &KContext = FContext.kernelContextMap.at(KName);

      auto &BInfo = KContext.blockIdxInfo;
      auto &OBs   = KContext.otherBlocks;

      if (VI >= BInfo.size())
        continue;

      IsAllVisited = false;
      if (VI == 0 && KI == 0) { // first if case
        VarStream << "  ";
      } else {
        VarStream << "  else ";
      }

      VarStream << "if " << PrintInfoToCondFunc("blockIdx.x", BInfo[VI]) << "\n"
                << "  {\n"
                << "    gridDim_x_ = " << KInfo.gridDim.size() << ";\n"
                << "    Others_    = " << OBs[VI] << ";\n"
                << "    KernelID_  = " << KI << ";\n"
                << "  }\n";
    }
  }
  VarStream << "  blockIdx_x_ = blockIdx.x - Others_;\n"
            << "  \n";
  VarStream.flush();

  return AnalysisContext{move(Kernels), move(ThreadNumMap), move(BranchConditionMap),
                         move(TmpBlockInfoString), move(NewBlockInfoString), move(NewFuncName), MaxThreadBound};
}
//---------------------------------------------------------------------------
} // namespace contexts
} // namespace bfuse
//---------------------------------------------------------------------------