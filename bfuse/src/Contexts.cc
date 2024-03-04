
#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include <map>

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Algorithms.h"
#include "bfuse/Contexts.h"

using namespace std;
//---------------------------------------------------------------------------
namespace bfuse {
namespace contexts {
//---------------------------------------------------------------------------
void KernelInfo::print(const string& KName) const
{
  cout << KName << ":\n";
  cout << "  KernelName: " << kernelName << "\n";
  cout << "  Barriers: " << hasBarriers << "\n";
  cout << "  GridDim:\n";
  cout << "    X: " << gridDim.x << "\n";
  cout << "    Y: " << gridDim.y << "\n";
  cout << "    Z: " << gridDim.z << "\n";
  cout << "  BlockDim:\n";
  cout << "    X: " << blockDim.x << "\n";
  cout << "    Y: " << blockDim.y << "\n";
  cout << "    Z: " << blockDim.z << "\n";
  cout << "  Reg: " << reg << "\n";
  cout << "  ExecTime: " << execTime << "\n";
}
//---------------------------------------------------------------------------
void FusionInfo::print() const
{
  cout << "[FusionInfo]\n";
  cout << "- File: " << filePath << "\n";
  cout << "  Kernels:\n";
  for (auto& KName : kernels) {
    cout << "   - " << KName << "\n";
  }
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

  // Intialize containers
  for (auto& KName : FInfo.kernels) {
    auto& Info = KInfoMap.at(KName);

    Kernels.push_back(KName);
    KernelInfoMap[KName] = Info;
  }
  // auto KernelContextMap = algorithms::zigZagBlockPattern(Kernels, KInfoMap);
  auto KernelContextMap = algorithms::sequentialBlockPattern(Kernels, KInfoMap);

  return FusionContext{move(Kernels), move(KernelInfoMap), move(KernelContextMap)};
}
//---------------------------------------------------------------------------
AnalysisContext AnalysisContext::create(FusionContext &FContext)
{
  // Lambda function
  // auto PrintInfoToCondFunc = [](string V, auto &Info) {
  //   string Str;
  //   llvm::raw_string_ostream RawStream{Str};
  //   RawStream << "(" << V << " >= " << Info.first << " && "
  //             << V << " < " << Info.second << ")";
  //   RawStream.flush();
  //   return Str;
  // };
  auto PrintInfoToCondFunc = [](string V, int bound) {
    string Str;
    llvm::raw_string_ostream RawStream{Str};
    RawStream << "(" << V << " < " << bound << ")";
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

  // 2. Initialize BranchConditionMap
  // -----------------------------------------------------------------
  // TODO: Add comments

  for (long unsigned I = 0; I < FContext.kernels.size(); ++I) {
    auto &KName    = FContext.kernels[I];
    auto &KInfo    = FContext.kernelInfoMap.at(KName);
    auto &KContext = FContext.kernelContextMap.at(KName);
    auto &TInfo    = KContext.threadIdxInfo;

    string CondStr;
    llvm::raw_string_ostream CondStream{CondStr};

    // "KernelID_" condition check
    CondStream << "(KernelID_ == " << I << ") && ";

    // threadIdx condition check
    // string CurrentThreadIdx = "(int)(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";
    string CurrentThreadIdx = "(int)threadIdx.x";
    // CondStream << "(" << PrintInfoToCondFunc(CurrentThreadIdx, TInfo) << ")";
    CondStream << PrintInfoToCondFunc(CurrentThreadIdx, KInfo.blockDim.size());
    // CondStream << PrintInfoToCondFunc("threadIdx.x", KInfo.blockDim.x) << " && "
    //            << PrintInfoToCondFunc("threadIdx.y", KInfo.blockDim.y) << " && "
    //            << PrintInfoToCondFunc("threadIdx.z", KInfo.blockDim.z);
    CondStream.flush();

    BranchConditionMap[KName] = CondStr;
  }

  // 3. Initialize TmpBlockInfoString
  // -----------------------------------------------------------------
  // TODO: Add comments

  llvm::raw_string_ostream TmpVarStream{TmpBlockInfoString};
  TmpVarStream << "\n"
               << "  // Temp declaration to avoid semantic errors\n"
               << "  int gridDim_x_  = 0;\n"
               << "  int gridDim_y_  = 0;\n"
               << "  int gridDim_z_  = 0;\n"
               << "  int blockIdx_x_ = 0;\n"
               << "  int blockIdx_y_ = 0;\n"
               << "  int blockIdx_z_ = 0;\n"
               << "  int blockDim_x_  = 0;\n"
               << "  int blockDim_y_  = 0;\n"
               << "  int blockDim_z_  = 0;\n"
               << "  int threadIdx_x_ = 0;\n"
               << "  int threadIdx_y_ = 0;\n"
               << "  int threadIdx_z_ = 0;\n";
  TmpVarStream.flush();

  // 4. Initialize NewBlockInfoString
  // -----------------------------------------------------------------
  // TODO: Add comments

  // NewBlockInfoString = algorithms::oldInterleaveBlockPattern(FContext);
  // NewBlockInfoString = algorithms::newInterleaveBlockPattern(FContext);
  // NewBlockInfoString = algorithms::advancedInterleaveBlockPattern(FContext);
  NewBlockInfoString = algorithms::advancedInterleaveBlockPattern2(FContext);

  return AnalysisContext{move(Kernels), move(ThreadNumMap), move(BranchConditionMap),
                         move(TmpBlockInfoString), move(NewBlockInfoString), move(NewFuncName), MaxThreadBound};
}
//---------------------------------------------------------------------------
} // namespace contexts
} // namespace bfuse
//---------------------------------------------------------------------------