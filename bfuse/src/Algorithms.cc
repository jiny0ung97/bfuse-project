
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
  // old version
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

  // Intialize containers & Calculate blockIdx boundary
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
string oldInterleaveBlockPattern(FusionContext &FContext)
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

  string NewBlockInfoString;
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
  VarStream << "  int gridDim_x_, gridDim_y_, gridDim_z_;\n"
            << "  int blockIdx_x_, blockIdx_y_, blockIdx_z_;\n"
            << "  int NewBlockIdx_;\n"
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

      string CurrentBlockIdx = "blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y";
      VarStream << "if " << PrintInfoToCondFunc(CurrentBlockIdx, BInfo[VI]) << "\n"
                << "  {\n"
                << "    NewBlockIdx_ = " << CurrentBlockIdx << " - " << OBs[VI] << ";\n"
                << "    KernelID_  = " << KI << ";\n"
                << "    gridDim_x_ = " << KInfo.gridDim.x << ";\n"
                << "    gridDim_y_ = " << KInfo.gridDim.y << ";\n"
                << "    gridDim_z_ = " << KInfo.gridDim.z << ";\n"
                << "  }\n";
    }
  }
  VarStream << "  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;\n"
            << "  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;\n"
            << "  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);\n";
  VarStream.flush();

  return NewBlockInfoString;
}
//---------------------------------------------------------------------------
string newInterleaveBlockPattern(FusionContext &FContext)
{
  // Lambda function
  auto PrintPairCondFunc = [](string V, int Bstart, int Bend) {
    string Str;
    llvm::raw_string_ostream RawStream{Str};
    RawStream << "(" << V << " >= " << Bstart << " && "
              << V << " < " << Bend << ")";
    RawStream.flush();
    return Str;
  };

  auto compareKernelsByThreadBlock = [&FContext](string KName1, string KName2) {
    auto& KInfo1 = FContext.kernelInfoMap.at(KName1);
    auto& KInfo2 = FContext.kernelInfoMap.at(KName2);

    return KInfo1.gridDim.size() < KInfo2.gridDim.size();
  };

  // Sorting kernels by Thread Blocks in ascending order
  vector<string> SortedKernels = FContext.kernels;
  sort(SortedKernels.begin(), SortedKernels.end(), compareKernelsByThreadBlock);

  // Calculate T(I)
  // T(0) = 0,
  // T(1) = (TB(1) - TB(1) % TotalSM) * TotalKernels
  // T(I) = T(I-1) + ((TB(I-1) - TB(I-1) % TotalSM) - (TB(I-2) - TB(I-2) % TotalSM)) * (TotalKernels - (I-1))
  vector<int> T(SortedKernels.size() + 1);
  int TotalSM = 84; // V100

  T[0] = 0;

  string &KName = SortedKernels[0];
  auto &KInfo   = FContext.kernelInfoMap.at(KName);
  T[1] = (KInfo.gridDim.size() - KInfo.gridDim.size() % TotalSM) * SortedKernels.size();

  for (long unsigned I = 2; I <= SortedKernels.size(); ++I) {
    string &KName1 = SortedKernels[I-1];
    string &KName2 = SortedKernels[I-2];
    auto &KInfo1   = FContext.kernelInfoMap.at(KName1);
    auto &KInfo2   = FContext.kernelInfoMap.at(KName2);

    T[I] = T[I-1] + ((KInfo1.gridDim.size() - KInfo1.gridDim.size() % TotalSM)
                     - (KInfo2.gridDim.size() - KInfo2.gridDim.size() % TotalSM))
                    * (SortedKernels.size() - (I-1));
  }

  auto PrintCondFunc = [&T, &SortedKernels, TotalSM](string V, long unsigned I, long unsigned J) {
    string Str;
    llvm::raw_string_ostream RawStream{Str};
    RawStream << "((" << V << " >= " << T[J] << " && "
              << V << " < " << T[J+1] << ")"
              << " && (((" << V << " - " << T[J] << ") / " << TotalSM << ") % "<< SortedKernels.size() - J << " == " << I - J << ")"
              << ")";
    RawStream.flush();
    return Str;
  };
  // string CurrentBlockIdx = "(int)(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)";
  string CurrentBlockIdx  = "(int)blockIdx.x";
  string CurrentThreadIdx = "(int)threadIdx.x";

  string NewBlockInfoString;
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
  VarStream << "  int gridDim_x_, gridDim_y_, gridDim_z_;\n"
            << "  int blockIdx_x_, blockIdx_y_, blockIdx_z_;\n"
            << "  int blockDim_x_, blockDim_y_, blockDim_z_;\n"
            << "  int threadIdx_x_, threadIdx_y_, threadIdx_z_;\n"
            << "  int NewBlockIdx_;\n"
            << "  int KernelID_;\n"
            << "  \n";

  // Condition 1 :
  //     And (0 <= J <= I)
  //                  T(J) <= blockIdx < T(J+1) && ((blockIdx - T(J)) / TotalSM) % (TotalKernels - J) == I - J
  for (long unsigned I = 0; I < SortedKernels.size(); ++I) {
    string &KName = SortedKernels[I];
    auto &KInfo   = FContext.kernelInfoMap.at(KName);

    for (long unsigned J = 0; J <= I; ++J) {
      if (I == 0) { // first if case
        VarStream << "  ";
      } else {
        VarStream << "  else ";
      }
      
      if (T[J] == T[J+1])
        continue;

      long unsigned KI = find(FContext.kernels.begin(), FContext.kernels.end(), KName) - FContext.kernels.begin();

      // BeforeBlocks = J == 0 ? 0 : TB(SortedKernels[J-1]) - TB(SortedKernels[J-1]) % TotalSM
      int BeforeBlocks = 0;
      if (J > 0) {
        string &KNameJ2 = SortedKernels[J-1];
        auto &KInfoJ2   = FContext.kernelInfoMap.at(KNameJ2);
        BeforeBlocks    = KInfoJ2.gridDim.size() - KInfoJ2.gridDim.size() % TotalSM;
      }

      // NewBlockIdx = BeforeBlocks + (blockIdx - T[J]) / (TotalSM * (TotalKernels - J)) * TotalSM + blockIdx % TotalSM
      VarStream << "if " << PrintCondFunc(CurrentBlockIdx, I, J) << "\n"
                << "  {\n"
                << "    NewBlockIdx_ = " << BeforeBlocks << " + "
                                         << "((" << CurrentBlockIdx << " - " << T[J] << ") / " << TotalSM * (SortedKernels.size() - J) << ") * " << TotalSM << " + "
                                         << CurrentBlockIdx << " % " << TotalSM << ";\n"
                << "    KernelID_  = " << KI << ";\n"
                << "    gridDim_x_ = " << KInfo.gridDim.x << ";\n"
                << "    gridDim_y_ = " << KInfo.gridDim.y << ";\n"
                << "    gridDim_z_ = " << KInfo.gridDim.z << ";\n"
                << "    blockDim_x_ = " << KInfo.blockDim.x << ";\n"
                << "    blockDim_y_ = " << KInfo.blockDim.y << ";\n"
                << "    blockDim_z_ = " << KInfo.blockDim.z << ";\n"
                << "  }\n";
    }
  }

  // Condition 2 :
  //      Sum (0 <= J < I)
  //                   TB(J) % TotalSM
  //      + T[TotalKernels]
  //      <= blockIdx <
  //      Sum (0 <= J <= I)
  //                   TB(J) % TotalSM
  //      + T[TotalKernels]
  for (long unsigned I = 0; I < SortedKernels.size(); ++I) {
    // Calculate accumulation of thread blocks
    int BlockSum = T[SortedKernels.size()];
    for (long unsigned J = 0; J < I; ++J) {
      string &KName = SortedKernels[J];
      auto &KInfo   = FContext.kernelInfoMap.at(KName);
      BlockSum += KInfo.gridDim.size() % TotalSM;
    }

    string &KName    = SortedKernels[I];
    auto &KInfo      = FContext.kernelInfoMap.at(KName);
    long unsigned KI = find(FContext.kernels.begin(), FContext.kernels.end(), KName) - FContext.kernels.begin();

    // NewBlockIdx = (TB(I) - TB(I) % TotalSM) + (blockIdx - BlockSum)
    //             = blockIdx - (BlockSum - (TB(I) - TB(I) % TotalSM))
    //             = blockIdx 
    VarStream << "  else if " << PrintPairCondFunc(CurrentBlockIdx, BlockSum, BlockSum + KInfo.gridDim.size() % TotalSM) << "\n"
              << "  {\n"
              << "    NewBlockIdx_ = " << CurrentBlockIdx << " - " << BlockSum - (KInfo.gridDim.size() - KInfo.gridDim.size() % TotalSM) << ";\n"
              << "    KernelID_  = " << KI << ";\n"
              << "    gridDim_x_ = " << KInfo.gridDim.x << ";\n"
              << "    gridDim_y_ = " << KInfo.gridDim.y << ";\n"
              << "    gridDim_z_ = " << KInfo.gridDim.z << ";\n"
              << "    blockDim_x_ = " << KInfo.blockDim.x << ";\n"
              << "    blockDim_y_ = " << KInfo.blockDim.y << ";\n"
              << "    blockDim_z_ = " << KInfo.blockDim.z << ";\n"
              << "  }\n";
  }

  VarStream << "  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;\n"
            << "  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;\n"
            << "  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);\n"
            << "  threadIdx_x_ = " << CurrentThreadIdx << " % blockDim_x_;\n"
            << "  threadIdx_y_ = " << CurrentThreadIdx << " / blockDim_x_ % blockDim_y_;\n"
            << "  threadIdx_z_ = " << CurrentThreadIdx << " / (blockDim_x_ * blockDim_y_);\n";
  VarStream.flush();

  return NewBlockInfoString;
}
//---------------------------------------------------------------------------
string advancedInterleaveBlockPattern(FusionContext &FContext)
{
  int TotalSMs = 84; // V100

  // gcd
  map<string, int> ThreadBlocks;
  int MaxBlockBound = 0;
  int GCD = 0;
  for (string &KName : FContext.kernels) {
    auto &KInfo         = FContext.kernelInfoMap.at(KName);
    ThreadBlocks[KName] = KInfo.gridDim.size() / TotalSMs;
    MaxBlockBound       += KInfo.gridDim.size() - KInfo.gridDim.size() % TotalSMs;

    if (KInfo.gridDim.size() / TotalSMs == 0)
      continue;
    
    if (GCD == 0) {
      GCD = KInfo.gridDim.size() / TotalSMs;
    } else {
      GCD = __gcd(GCD, KInfo.gridDim.size() / TotalSMs);
    }
  }

  map<string, int> AccTickets;
  int Acc = 0;
  for (string &KName : FContext.kernels) {
    ThreadBlocks[KName] = ThreadBlocks.at(KName) / GCD;
    AccTickets[KName]   = Acc;
    Acc += ThreadBlocks.at(KName);
  }

  // Thread Block boundary
  string NewBlockInfoString;
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
  VarStream << "  int gridDim_x_, gridDim_y_, gridDim_z_;\n"
            << "  int blockIdx_x_, blockIdx_y_, blockIdx_z_;\n"
            << "  int blockDim_x_, blockDim_y_, blockDim_z_;\n"
            << "  int threadIdx_x_, threadIdx_y_, threadIdx_z_;\n"
            << "  int NewBlockIdx_;\n"
            << "  int KernelID_;\n"
            << "  \n";

  string CurrentBlockIdx  = "(int)blockIdx.x";
  string CurrentThreadIdx = "(int)threadIdx.x";
  bool IsFirst = true;

  for (long unsigned KI = 0; KI < FContext.kernels.size(); ++KI) {
    string &KName = FContext.kernels[KI];
    auto &KInfo   = FContext.kernelInfoMap.at(KName);

    int Ticket = ThreadBlocks.at(KName);
    if (Ticket == 0)
      continue;
    
    VarStream << "  ";
    if (IsFirst) { // first if case
      IsFirst = false;
    } else {
      VarStream << "else ";
    }
    
    // RotateSize = Acc * TotalSMs
    // time = blockIdx / RotateSize
    // (blockIdx < MaxBlockBound) && (blockIdx % RotateSize / TotalSMs >= AccTickets.at(KName)) && (... < AccTickets.at(KName) + Ticket)
    VarStream << "if ("
              << "(" << CurrentBlockIdx << " < " << MaxBlockBound << ")"
              << " && (" << CurrentBlockIdx << " % " << Acc * TotalSMs << " / " << TotalSMs << " >= " << AccTickets.at(KName) << ")"
              << " && (" << CurrentBlockIdx << " % " << Acc * TotalSMs << " / " << TotalSMs << " < " << AccTickets.at(KName) + Ticket << ")"
              << ")\n";

    // NewBlockIdx = time * Tickets * TotalSMs + blockIdx % RotateSize - AccTickets.at(KName) * TotalSMS
    VarStream << "  {\n"
              << "    NewBlockIdx_ = " << CurrentBlockIdx << " / " << Acc * TotalSMs << " * " << Ticket * TotalSMs
              << " + " << CurrentBlockIdx << " % " << Acc * TotalSMs << " - " <<  AccTickets.at(KName) * TotalSMs << ";\n"
              << "    KernelID_  = " << KI << ";\n"
              << "    gridDim_x_ = " << KInfo.gridDim.x << ";\n"
              << "    gridDim_y_ = " << KInfo.gridDim.y << ";\n"
              << "    gridDim_z_ = " << KInfo.gridDim.z << ";\n"
              << "    blockDim_x_ = " << KInfo.blockDim.x << ";\n"
              << "    blockDim_y_ = " << KInfo.blockDim.y << ";\n"
              << "    blockDim_z_ = " << KInfo.blockDim.z << ";\n"
              << "  }\n";
  }

  int AccBound = MaxBlockBound;
  for (long unsigned KI = 0; KI < FContext.kernels.size(); ++KI) {
    string &KName = FContext.kernels[KI];
    auto &KInfo   = FContext.kernelInfoMap.at(KName);

    VarStream << "  ";
    if (IsFirst) { // first if case
      IsFirst = false;
    } else {
      VarStream << "else ";
    }

    // (blockIdx >= AccBound) && (blockIdx < AccBound + KInfo.gridDim.size() % TotalSMs)
    VarStream << "if ("
              << "(" << CurrentBlockIdx << " >= " << AccBound << ")"
              << " && (" << CurrentBlockIdx << " < " << AccBound + KInfo.gridDim.size() % TotalSMs << ")"
              << ")\n";

    // NewBlockIdx_ = blockIdx - AccBound + KInfo.gridDim.size() - KInfo.gridDim.size() % TotalSMs
    VarStream << "  {\n"
              << "    NewBlockIdx_ = " << CurrentBlockIdx << " - " << AccBound - KInfo.gridDim.size() + KInfo.gridDim.size() % TotalSMs << ";\n"
              << "    KernelID_  = " << KI << ";\n"
              << "    gridDim_x_ = " << KInfo.gridDim.x << ";\n"
              << "    gridDim_y_ = " << KInfo.gridDim.y << ";\n"
              << "    gridDim_z_ = " << KInfo.gridDim.z << ";\n"
              << "    blockDim_x_ = " << KInfo.blockDim.x << ";\n"
              << "    blockDim_y_ = " << KInfo.blockDim.y << ";\n"
              << "    blockDim_z_ = " << KInfo.blockDim.z << ";\n"
              << "  }\n";

    AccBound += KInfo.gridDim.size() % TotalSMs;
  }

  VarStream << "  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;\n"
            << "  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;\n"
            << "  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);\n"
            << "  threadIdx_x_ = " << CurrentThreadIdx << " % blockDim_x_;\n"
            << "  threadIdx_y_ = " << CurrentThreadIdx << " / blockDim_x_ % blockDim_y_;\n"
            << "  threadIdx_z_ = " << CurrentThreadIdx << " / (blockDim_x_ * blockDim_y_);\n";
  VarStream.flush();

  return NewBlockInfoString;
}
//---------------------------------------------------------------------------
string advancedInterleaveBlockPattern2(FusionContext &FContext)
{
  // Normalize
  map<string, int> BlockRatio;
  float Min    = -1;
  int TotalSMs = 84; // V100

  for (string &KName : FContext.kernels) {
    auto &KInfo    = FContext.kernelInfoMap.at(KName);
    float ExecTime = KInfo.execTime == -1 ? 1 : KInfo.execTime;
    float Ratio    = KInfo.gridDim.size() * ExecTime;

    if (Min == -1) {
      Min = Ratio;
    } else {
      Min = Min < Ratio ? Min : Ratio;
    }
  }

  map<string, int> AccTickets;
  int CurTickets = 0;
  for (string &KName : FContext.kernels) {
    auto &KInfo    = FContext.kernelInfoMap.at(KName);
    float ExecTime = KInfo.execTime == -1 ? 1 : KInfo.execTime;
    float Ratio    = KInfo.gridDim.size() * ExecTime;

    BlockRatio[KName] = (int)(Ratio / Min);
    AccTickets[KName] = CurTickets;
    CurTickets += (int)(Ratio / Min);
  }

  // Calculate real max thread block bound
  map<string, int> RealMaxBlockBound;
  map<string, int> RemainThreadBlocks;
  map<string, bool> IsReached;
  int RotateSize = 0;
  int RotateNum  = 0;
  bool IsRemain = true;

  for (string &KName : FContext.kernels) {
    auto &KInfo = FContext.kernelInfoMap.at(KName);
    auto &Ratio = BlockRatio.at(KName);

    RealMaxBlockBound[KName]  = 0;
    RemainThreadBlocks[KName] = KInfo.gridDim.size();
    IsReached[KName] = false;
    RotateSize += TotalSMs * Ratio;
  }

  while (IsRemain) {
    IsRemain = false;

    for (string &KName : FContext.kernels) {
      auto &KInfo   = FContext.kernelInfoMap.at(KName);
      auto &Ratio   = BlockRatio.at(KName);
      auto &Acc     = AccTickets.at(KName);
      auto &Bound   = RealMaxBlockBound.at(KName);
      auto &Remain  = RemainThreadBlocks.at(KName);
      auto &Reached = IsReached.at(KName);

      int RotateTBs;
      int TBs;

      if (Reached)
        continue;

      if (Remain > TotalSMs * Ratio) {
        RotateTBs = RotateSize;
        TBs       = TotalSMs * Ratio;
      } else {
        RotateTBs = Acc * TotalSMs + Remain;
        TBs       = Remain;
        Reached   = true;
      }

      Bound   += RotateTBs;
      Remain  -= TBs;
      IsRemain = true;
    }
    RotateNum += 1;
  }

  // Thread Block if branch
  string NewBlockInfoString;
  llvm::raw_string_ostream VarStream{NewBlockInfoString};
  
  // Comments
  VarStream << "  /*\n"
            << "   * KernelID_ means...\n";
  for (long unsigned I = 0; I < FContext.kernels.size(); ++I) {
    string &KName = FContext.kernels[I];
    VarStream << "   * " << I << ": " << KName << "\n";
  }
  VarStream << "   * Kernel's Thread Blocks are " << RotateNum * RotateSize << "\n";
  VarStream << "   */\n";

  // Declarations
  VarStream << "  int gridDim_x_, gridDim_y_, gridDim_z_;\n"
            << "  int blockIdx_x_, blockIdx_y_, blockIdx_z_;\n"
            << "  int blockDim_x_, blockDim_y_, blockDim_z_;\n"
            << "  int threadIdx_x_, threadIdx_y_, threadIdx_z_;\n"
            << "  int NewBlockIdx_;\n"
            << "  int KernelID_ = -1;\n"
            << "  \n";

  string CurrentBlockIdx  = "(int)blockIdx.x";
  string CurrentThreadIdx = "(int)threadIdx.x";
  bool IsFirst = true;

  for (long unsigned KI = 0; KI < FContext.kernels.size(); ++KI) {
    string &KName  = FContext.kernels[KI];
    auto &KInfo    = FContext.kernelInfoMap.at(KName);
    auto &MaxBound = RealMaxBlockBound.at(KName);
    auto &Ticket   = BlockRatio.at(KName);
    auto &Acc      = AccTickets.at(KName);
    
    VarStream << "  ";
    if (IsFirst) { // first if case
      IsFirst = false;
    } else {
      VarStream << "else ";
    }

    // Ticket = BlockRatio
    // (blockIDx < RealMaxBlockBound) && (blockIdx % RotateSize / TotalSMs >= AccTickets) && (... < AccTickets + Ticket)
    VarStream << "if ("
              << "(" << CurrentBlockIdx << " < " << MaxBound << ")"
              << " && (" << CurrentBlockIdx << " % " << RotateSize << " / " << TotalSMs << " >= " << Acc << ")"
              << " && (" << CurrentBlockIdx << " % " << RotateSize << " / " << TotalSMs << " < " << Acc + Ticket << ")"
              << ")\n";
              
    // NewBlockIdx_ = (blockIdx / RotateSize) * Ticket * TotalSMs + (blockIdx % RotateSize - AccTickets * TotalSMs)
    VarStream << "  {\n"
              << "    NewBlockIdx_ = " << "(" << CurrentBlockIdx << " / " << RotateSize << ") * " << Ticket * TotalSMs
              << " + " << CurrentBlockIdx << " % " << RotateSize << " - " << Acc * TotalSMs << ";\n"
              << "    KernelID_  = " << KI << ";\n"
              << "    gridDim_x_ = " << KInfo.gridDim.x << ";\n"
              << "    gridDim_y_ = " << KInfo.gridDim.y << ";\n"
              << "    gridDim_z_ = " << KInfo.gridDim.z << ";\n"
              << "    blockDim_x_ = " << KInfo.blockDim.x << ";\n"
              << "    blockDim_y_ = " << KInfo.blockDim.y << ";\n"
              << "    blockDim_z_ = " << KInfo.blockDim.z << ";\n"
              << "  }\n";
  }

  VarStream << "  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;\n"
            << "  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;\n"
            << "  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);\n"
            << "  threadIdx_x_ = " << CurrentThreadIdx << " % blockDim_x_;\n"
            << "  threadIdx_y_ = " << CurrentThreadIdx << " / blockDim_x_ % blockDim_y_;\n"
            << "  threadIdx_z_ = " << CurrentThreadIdx << " / (blockDim_x_ * blockDim_y_);\n";
  VarStream.flush();

  return NewBlockInfoString;
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
      if (ShrdVarListMap.find(KName) == ShrdVarListMap.end())
        continue;
        
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