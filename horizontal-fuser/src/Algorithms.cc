
#include <algorithm>
#include <vector>
#include <map>
#include <tuple>

#include "llvm/Support/raw_ostream.h"

#include "fuse/Contexts.h"
#include "fuse/Algorithms.h"

using namespace std;

using namespace fuse::contexts;
//---------------------------------------------------------------------------
namespace fuse {
namespace algorithms {
//---------------------------------------------------------------------------
tuple<string, map<string, string>, GridDim, BlockDim> fineInterleavePattern(vector<string> &Kernels, map<string, KernelInfo> &KernelInfoMap, int TotalSM)
{
  auto compareKernelsByThreadBlock = [&KernelInfoMap](string KName1, string KName2) {
    auto& KInfo1 = KernelInfoMap.at(KName1);
    auto& KInfo2 = KernelInfoMap.at(KName2);

    return KInfo1.GridDim_.size() < KInfo2.GridDim_.size();
  };

  // Sorting kernels by Thread Blocks in ascending order
  vector<string> SortedKernels = Kernels;
  sort(SortedKernels.begin(), SortedKernels.end(), compareKernelsByThreadBlock);

  vector<pair<int, int>> BlockBounds;
  vector<int> MyBlocks;
  map<string, int> RemainBounds;
  int EndBounds = 0;
  int MyBounds  = 0;
  int AccBounds = 0;

  for (long unsigned I = 0; I < SortedKernels.size(); ++I) {
     string &KName = SortedKernels[I];
     auto &KInfo   = KernelInfoMap.at(KName);

     int CurTotalKNum = SortedKernels.size() - I;
     auto Bound       = make_pair(EndBounds, AccBounds + (KInfo.GridDim_.size() / TotalSM) * TotalSM * CurTotalKNum);

     BlockBounds.push_back(Bound);
     MyBlocks.push_back(MyBounds);
     RemainBounds[KName] = int(KInfo.GridDim_.size() % TotalSM);

     EndBounds = Bound.second;
     MyBounds  = int(KInfo.GridDim_.size() / TotalSM) * TotalSM;
     AccBounds += int(KInfo.GridDim_.size() / TotalSM) * TotalSM;
  }

  map<string, pair<int, int>> RemainBlockBounds;
  for (auto &KName : Kernels) {
    int Remains = RemainBounds.at(KName);
    auto Bound  = make_pair(EndBounds, EndBounds + Remains);

    RemainBlockBounds[KName] = Bound;
    EndBounds                = Bound.second;
  }

  // Print Condition
  string NewBlockInfoStr;
  llvm::raw_string_ostream VarStream{NewBlockInfoStr};

  auto PrintPairCondFunc = [=](string V, int BStart, int BEnd, long unsigned CurTotalKNum, long unsigned CurKernelIdx) {
    string Str;
    llvm::raw_string_ostream RawStream{Str};

    RawStream << "("
              << "(" << V << " >= " << BStart << " && " << V << " < " << BEnd << ")"
              << " && "
              << "(int((" << V << " - " << BStart << ") % " << TotalSM * CurTotalKNum << " / " << TotalSM << ") == " << CurKernelIdx << ")"
              << ")";
    RawStream.flush();
    return Str;
  };

  VarStream << "  /*\n"
            << "   * KernelID_ means...\n";
  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    string &KName = Kernels[I];
    VarStream << "   * " << I << ": " << KName << "\n";
  }
  VarStream << "   */\n";
  VarStream << "  int gridDim_x_, gridDim_y_, gridDim_z_;\n"
            << "  int blockIdx_x_, blockIdx_y_, blockIdx_z_;\n"
            << "  int blockDim_x_, blockDim_y_, blockDim_z_;\n"
            << "  int threadIdx_x_, threadIdx_y_, threadIdx_z_;\n"
            << "  int NewBlockIdx_;\n"
            << "  int KernelID_;\n"
            << "  \n";

  // string CurrentBlockIdx  = "(int)(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)";
  // string CurrentThreadIdx = "(int)(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";
  string CurrentBlockIdx  = "(int)blockIdx.x";
  string CurrentThreadIdx = "(int)threadIdx.x";
  bool FirstIf = true;

  for (long unsigned I = 0; I < SortedKernels.size(); ++I) {
    for (long unsigned J = I; J < SortedKernels.size(); ++J) {
      auto &KName  = SortedKernels[J];
      auto &KInfo  = KernelInfoMap.at(KName);
      auto &Bounds = BlockBounds[I];

      long unsigned CurTotalNum  = SortedKernels.size() - I;
      long unsigned CurKernelIdx = J - I;
      long unsigned KI           = find(Kernels.begin(), Kernels.end(), KName) - Kernels.begin();

      if (FirstIf) {
        VarStream << "  if " <<  PrintPairCondFunc(CurrentBlockIdx, Bounds.first, Bounds.second, CurTotalNum, CurKernelIdx) << "\n";
        FirstIf = false;
      } else {
        VarStream << "  else if " <<  PrintPairCondFunc(CurrentBlockIdx, Bounds.first, Bounds.second, CurTotalNum, CurKernelIdx) << "\n";
      }
      VarStream << "  {\n"
                << "    NewBlockIdx_ = " << MyBlocks[I]
                                         << " + (" << CurrentBlockIdx << " - " << Bounds.first << ") % " << TotalSM
                                         << " + int((" << CurrentBlockIdx << " - " << Bounds.first << ") / " << CurTotalNum * TotalSM << ") * " << TotalSM << ";\n"
                << "    KernelID_  = " << KI << ";\n"
                << "    gridDim_x_ = " << KInfo.GridDim_.X << ";\n"
                << "    gridDim_y_ = " << KInfo.GridDim_.Y << ";\n"
                << "    gridDim_z_ = " << KInfo.GridDim_.Z << ";\n"
                << "    blockDim_x_ = " << KInfo.BlockDim_.X << ";\n"
                << "    blockDim_y_ = " << KInfo.BlockDim_.Y << ";\n"
                << "    blockDim_z_ = " << KInfo.BlockDim_.Z << ";\n"
                << "  }\n";
    }
  }

  auto PrintPairCondFunc2 = [](string V, int Bstart, int Bend) {
    string Str;
    llvm::raw_string_ostream RawStream{Str};
    RawStream << "(" << V << " >= " << Bstart << " && " << V << " < " << Bend << ")";
    RawStream.flush();
    return Str;
  };

  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    auto &KName  = Kernels[I];
    auto &KInfo  = KernelInfoMap.at(KName);
    auto &Bounds = RemainBlockBounds.at(KName);
    
    if (FirstIf) {
      VarStream << "  if " << PrintPairCondFunc2(CurrentBlockIdx, Bounds.first, Bounds.second) << "\n";
      FirstIf = false;
    } else {
      VarStream << "  else if " << PrintPairCondFunc2(CurrentBlockIdx, Bounds.first, Bounds.second) << "\n";
    }
    VarStream << "  {\n"
              << "    NewBlockIdx_ = " << CurrentBlockIdx << " - " << Bounds.first << " + " << KInfo.GridDim_.size() - KInfo.GridDim_.size() % TotalSM << ";\n"
              << "    KernelID_  = " << I << ";\n"
              << "    gridDim_x_ = " << KInfo.GridDim_.X << ";\n"
              << "    gridDim_y_ = " << KInfo.GridDim_.Y << ";\n"
              << "    gridDim_z_ = " << KInfo.GridDim_.Z << ";\n"
              << "    blockDim_x_ = " << KInfo.BlockDim_.X << ";\n"
              << "    blockDim_y_ = " << KInfo.BlockDim_.Y << ";\n"
              << "    blockDim_z_ = " << KInfo.BlockDim_.Z << ";\n"
              << "  }\n";
  }

  VarStream << "  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;\n"
            << "  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;\n"
            << "  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);\n"
            << "  threadIdx_x_ = " << CurrentThreadIdx << " % blockDim_x_;\n"
            << "  threadIdx_y_ = " << CurrentThreadIdx << " / blockDim_x_ % blockDim_y_;\n"
            << "  threadIdx_z_ = " << CurrentThreadIdx << " / (blockDim_x_ * blockDim_y_);\n";
  VarStream.flush();

  // Print thread condition
  map<string, string> NewCondStrMap;
  int IsFirstCond = true;

  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    string NewCondStr;
    llvm::raw_string_ostream CondStream{NewCondStr};

    auto &KName = Kernels[I];
    auto &KInfo = KernelInfoMap.at(KName);

    if (IsFirstCond) {
      CondStream << "  if (KernelID_ == " << I << ")"
                 << " && (" << CurrentThreadIdx << " >= 0 && " << CurrentThreadIdx << " < " << KInfo.BlockDim_.size() << ")\n";
      IsFirstCond = false;
    } else {
      CondStream << "  else if (KernelID_ == " << I << ")"
                 << " && (" << CurrentThreadIdx << " >= 0 && " << CurrentThreadIdx << " < " << KInfo.BlockDim_.size() << ")\n";
    }

    CondStream.flush();
    NewCondStrMap[KName] = NewCondStr;
  }

  int AccBlocks  = 0;
  int MaxThreads = 0;
  for (auto &KName : Kernels) {
    auto &KInfo   = KernelInfoMap.at(KName);
    AccBlocks     += KInfo.GridDim_.size();
    MaxThreads    = MaxThreads > KInfo.BlockDim_.size() ? MaxThreads : KInfo.BlockDim_.size();
  }

  // GridDim, BlockDim
  GridDim FusedGridDim;
  BlockDim FusedBlockDim;

  FusedGridDim.X = AccBlocks;
  FusedGridDim.Y = 1;
  FusedGridDim.Z = 1;
  FusedBlockDim.X = MaxThreads;
  FusedBlockDim.Y = 1;
  FusedBlockDim.Z = 1;

  return make_tuple(NewBlockInfoStr, NewCondStrMap, FusedGridDim, FusedBlockDim);
}
//---------------------------------------------------------------------------
} // namespace algorithms
} // namespace fuse
//---------------------------------------------------------------------------