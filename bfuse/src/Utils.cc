
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "bfuse/Bfuse.h"
#include "bfuse/Utils.h"

using namespace std;
//---------------------------------------------------------------------------
namespace bfuse {
namespace utils {
//---------------------------------------------------------------------------
void printFusionInfo(const FusionInfo& Info)
{
  cout << "> FusionInfo\n";
  cout << "  - Kernels:\n";
  for (auto& KName : Info.kernels) {
    cout << "    - " << KName << "\n";
  }
}
//---------------------------------------------------------------------------
void printKernelInfo(const string& KName, const KernelInfo& Info)
{
  cout << "[KernelInfo]\n";
  cout << KName << "\n";
  cout << "  File: " << Info.filePath << "\n";
  cout << "  Barriers: " << Info.hasBarriers << "\n";
  cout << "  GridDim:\n";
  cout << "    X: " << Info.gridDim.x << "\n";
  cout << "    Y: " << Info.gridDim.y << "\n";
  cout << "    Z: " << Info.gridDim.z << "\n";
  cout << "  BlockDim:\n";
  cout << "    X: " << Info.blockDim.x << "\n";
  cout << "    Y: " << Info.blockDim.y << "\n";
  cout << "    Z: " << Info.blockDim.z << "\n";
}
//---------------------------------------------------------------------------
void printKernelContexts(const string& KName, const tools::KernelContext& Context)
{
    auto& ThreadIdxInfo = Context.threadIdxInfo;
    auto& BlockIdxInfo = Context.blockIdxInfo;
    auto& OtherBlocks = Context.otherBlocks;

    cout << "[KernelContext Info]\n";
    cout << "  ThreadIdxInfo: [" << ThreadIdxInfo.first << " ~ " << ThreadIdxInfo.second << ")\n";
    cout << "  BlockIdxInfo: ";
    for (auto& info : BlockIdxInfo) {
      cout << "[" << info.first << " ~ " << info.second << ") ";
    }
    cout << "\n";
    cout << "  OtherBlocks: ";
    for (auto n : OtherBlocks) {
      cout << n << " ";
    }
    cout << "\n";
}
//---------------------------------------------------------------------------
void printFusionTools(const tools::FusionTools& Tools, const FusionInfo& Info)
{
  cout << "\n================= FusionTools =================";
  for (auto& KName : Info.kernels) {
    auto KernelInfo    = Tools.getKernelInfo(KName);
    auto KernelContext = Tools.getKernelContext(KName);

    cout << "\n//// Tool Objectes ////\n";
    printKernelInfo(KName, KernelInfo);
    cout << "\n";
    printKernelContexts(KName, KernelContext);
  }
}
//---------------------------------------------------------------------------
} // namespace utils
} // namespace bfuse
//---------------------------------------------------------------------------