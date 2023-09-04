
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "bfuse.h"
#include "utils.h"

using namespace std;
//---------------------------------------------------------------------------
namespace bfuse {
namespace utils {
//---------------------------------------------------------------------------
void printFusionYAML(const vector<FusionInfo>& Infos)
{
  cout << "\n========= Fusion Info =========";
  for (auto& info : Infos) {
    cout << "\n- File: " << info.filePath << "\n";
    cout << "- Kernels:\n";
    for (auto& kernel : info.kernels) {
      cout << "  + " << kernel << "\n";
    }
  }
}
//---------------------------------------------------------------------------
void printKernelYAML(const map<std::string, KernelInfo>& Infos)
{
  cout << "\n========= Kernel Info =========";
  for (auto& info : Infos) {
    cout << "\n<" << info.first << ">\n";
    cout << "- Name: " << info.second.kernelName << "\n";
    cout << "- Barriers: " << info.second.hasBarriers << "\n";
    cout << "- GridDim: " << info.second.gridDim.size() << "\n";
    cout << "- BlockDim: " << info.second.blockDim.size() << "\n";
    cout << "- Registers: " << info.second.reg << "\n";
    cout << "- ExecTime: " << info.second.execTime << "\n";
  }
}
//---------------------------------------------------------------------------
void printKernelContexts(const vector<tools::KernelContext>& Contexts)
{
  cout << "\n========= Fusion's Context Info =========";
  for (auto& Context : Contexts) {
    cout << "\n> KernelContext Info\n";

    auto& Info = Context.info;
    cout << "  <Kernel Info>\n";
    cout << "  - Name: " << Info.kernelName << "\n";
    cout << "  - Barriers: " << Info.hasBarriers << "\n";
    cout << "  - GridDim: " << Info.gridDim.size() << "\n";
    cout << "  - BlockDim: " << Info.blockDim.size() << "\n";
    cout << "  - Registers: " << Info.reg << "\n";
    cout << "  - ExecTime: " << Info.execTime << "\n";

    auto& ThreadIdxInfo = Context.threadIdxInfo;
    cout << "  <ThreadIdx Info>\n";
    cout << "  - Boundary:\n";
    cout << "              [" << ThreadIdxInfo.first << " ~ " << ThreadIdxInfo.second << ")\n";

    auto& BlockIdxInfo = Context.blockIdxInfo;
    cout << "  <BlockIdx Info>\n";
    cout << "  - Boundary:\n";
    for (auto& info : BlockIdxInfo) {
      cout << "              [" << info.first << " ~ " << info.second << ")\n";
    }

    auto& OtherBlocks = Context.otherBlocks;
    cout << "  <Other Blocks>\n";
    cout << "  - Left: ";
    for (auto n : OtherBlocks) {
      cout << n << " ";
    }
    cout << "\n";
  }
}
//---------------------------------------------------------------------------
} // namespace utils
} // namespace bfuse
//---------------------------------------------------------------------------