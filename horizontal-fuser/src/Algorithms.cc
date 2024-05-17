
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
tuple<map<string, string>, map<string, string>, GridDim, BlockDim> hfusePattern(vector<string> &Kernels, map<string, KernelInfo> &KernelInfoMap)
{
  map<string, pair<int, int>> ThreadBounds;

  int AccBounds = 0;
  for (auto &KName : Kernels) {
    auto &KInfo = KernelInfoMap.at(KName);
    ThreadBounds[KName] = make_pair(AccBounds, AccBounds + KInfo.BlockDim_.size());

    int Threads;
    if (KInfo.BlockDim_.size() % 32 == 0) {
      Threads = KInfo.BlockDim_.size();
    } else {
      Threads = (int(KInfo.BlockDim_.size() / 32) + 1) * 32;
    }
    AccBounds += Threads;
  }

  // Print Condition
  auto PrintPairCondFunc = [=](string VT, int BStart, int BEnd, string VB, int BBEnd) {
    string Str;
    llvm::raw_string_ostream RawStream{Str};

    RawStream << "("
              << "(" << VT << " >= " << BStart << " && " << VT << " < " << BEnd << ")"
              << " && "
              << "(" << VB << " >= 0 && " << VB << " < " << BBEnd << ")"
              << ")";
    RawStream.flush();
    return Str;
  };

  // Generate CondStrMap
  map<string, string> CondStrMap;

  // string CurrentBlockIdx  = "(int)(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)";
  // string CurrentThreadIdx = "(int)(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";
  string CurrentBlockIdx  = "(int)blockIdx.x";
  string CurrentThreadIdx = "(int)threadIdx.x";
  bool FirstIf = true;

  for (auto &KName : Kernels) {
    string CondStr;
    llvm::raw_string_ostream CondStream{CondStr};

    auto &KInfo  = KernelInfoMap.at(KName);
    auto &Bounds = ThreadBounds.at(KName);

    if (FirstIf) {
      CondStream << "  if " << PrintPairCondFunc(CurrentThreadIdx, Bounds.first, Bounds.second, CurrentBlockIdx, KInfo.GridDim_.size()) << "\n";
    } else {
      CondStream << "  else if " << PrintPairCondFunc(CurrentThreadIdx, Bounds.first, Bounds.second, CurrentBlockIdx, KInfo.GridDim_.size()) << "\n";
    }
    CondStream.flush();
    CondStrMap[KName] = CondStr;
  }

  // Generate BlockDeclStrMap
  map<string, string> BlockDeclStrMap;

  for (auto &KName : Kernels) {
    auto &KInfo  = KernelInfoMap.at(KName);
    auto &Bounds = ThreadBounds.at(KName);

    string BlockDeclStr;
    llvm::raw_string_ostream DeclStream{BlockDeclStr};

    string NewTid = "(" + CurrentThreadIdx + " - " + std::to_string(Bounds.first) + ")";

    DeclStream << "      int blockIdx_x_ = " << CurrentBlockIdx << " % " << KInfo.GridDim_.X << ";\n"
               << "      int blockIdx_y_ = " << CurrentBlockIdx << " / " << KInfo.GridDim_.X << " % " << KInfo.GridDim_.Y << ";\n"
               << "      int blockIdx_z_ = " << CurrentBlockIdx << " / " << KInfo.GridDim_.X * KInfo.GridDim_.Y << ";\n"
               << "      int threadIdx_x_ = " << NewTid << " % " << KInfo.BlockDim_.X << ";\n"
               << "      int threadIdx_y_ = " << NewTid << " / " << KInfo.BlockDim_.X << " % " << KInfo.BlockDim_.Y << ";\n"
               << "      int threadIdx_z_ = " << NewTid << " / " << KInfo.BlockDim_.X * KInfo.BlockDim_.Y << ";\n";
    DeclStream.flush();
    BlockDeclStrMap[KName] = BlockDeclStr;
  }

  int MaxBlocks = 0;
  for (auto &KName : Kernels) {
    auto &KInfo = KernelInfoMap.at(KName);
    MaxBlocks   = MaxBlocks > KInfo.GridDim_.size() ? MaxBlocks : KInfo.GridDim_.size();
  }

  // Generate GridDim, BlockDim
  GridDim FusedGridDim;
  BlockDim FusedBlockDim;

  FusedGridDim.X = MaxBlocks;
  FusedGridDim.Y = 1;
  FusedGridDim.Z = 1;
  FusedBlockDim.X = AccBounds;
  FusedBlockDim.Y = 1;
  FusedBlockDim.Z = 1;

  return make_tuple(BlockDeclStrMap, CondStrMap, FusedGridDim, FusedBlockDim);
}
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

  // Generate BlockDeclStr
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

  string BlockDeclStr;
  llvm::raw_string_ostream DeclStream{BlockDeclStr};

  DeclStream << "  /*\n"
            << "   * KernelID_ means...\n";
  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    string &KName = Kernels[I];
    DeclStream << "   * " << I << ": " << KName << "\n";
  }
  DeclStream << "   */\n";
  DeclStream << "  int gridDim_x_, gridDim_y_, gridDim_z_;\n"
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
        DeclStream << "  if " <<  PrintPairCondFunc(CurrentBlockIdx, Bounds.first, Bounds.second, CurTotalNum, CurKernelIdx) << "\n";
        FirstIf = false;
      } else {
        DeclStream << "  else if " <<  PrintPairCondFunc(CurrentBlockIdx, Bounds.first, Bounds.second, CurTotalNum, CurKernelIdx) << "\n";
      }
      DeclStream << "  {\n"
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
      DeclStream << "  if " << PrintPairCondFunc2(CurrentBlockIdx, Bounds.first, Bounds.second) << "\n";
      FirstIf = false;
    } else {
      DeclStream << "  else if " << PrintPairCondFunc2(CurrentBlockIdx, Bounds.first, Bounds.second) << "\n";
    }
    DeclStream << "  {\n"
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

  DeclStream << "  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;\n"
             << "  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;\n"
             << "  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);\n"
             << "  threadIdx_x_ = " << CurrentThreadIdx << " % blockDim_x_;\n"
             << "  threadIdx_y_ = " << CurrentThreadIdx << " / blockDim_x_ % blockDim_y_;\n"
             << "  threadIdx_z_ = " << CurrentThreadIdx << " / (blockDim_x_ * blockDim_y_);\n";
  DeclStream.flush();

  // Generate CondStrMap
  map<string, string> CondStrMap;
  int IsFirstCond = true;

  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    string CondStr;
    llvm::raw_string_ostream CondStream{CondStr};

    auto &KName = Kernels[I];
    auto &KInfo = KernelInfoMap.at(KName);

    if (IsFirstCond) {
      CondStream << "  if ((KernelID_ == " << I << ")"
                 << " && (" << CurrentThreadIdx << " >= 0 && " << CurrentThreadIdx << " < " << KInfo.BlockDim_.size() << "))\n";
      IsFirstCond = false;
    } else {
      CondStream << "  else if ((KernelID_ == " << I << ")"
                 << " && (" << CurrentThreadIdx << " >= 0 && " << CurrentThreadIdx << " < " << KInfo.BlockDim_.size() << "))\n";
    }

    CondStream.flush();
    CondStrMap[KName] = CondStr;
  }

  int AccBlocks  = 0;
  int MaxThreads = 0;
  for (auto &KName : Kernels) {
    auto &KInfo   = KernelInfoMap.at(KName);
    AccBlocks     += KInfo.GridDim_.size();
    MaxThreads    = MaxThreads > KInfo.BlockDim_.size() ? MaxThreads : KInfo.BlockDim_.size();
  }

  // Generate GridDim, BlockDim
  GridDim FusedGridDim;
  BlockDim FusedBlockDim;

  FusedGridDim.X = AccBlocks;
  FusedGridDim.Y = 1;
  FusedGridDim.Z = 1;
  FusedBlockDim.X = MaxThreads;
  FusedBlockDim.Y = 1;
  FusedBlockDim.Z = 1;

  return make_tuple(BlockDeclStr, CondStrMap, FusedGridDim, FusedBlockDim);
}
//---------------------------------------------------------------------------
} // namespace algorithms
} // namespace fuse
//---------------------------------------------------------------------------