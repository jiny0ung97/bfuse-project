
#include <iostream>

#include "utils.h"

using namespace std;
using namespace bfuse;
//---------------------------------------------------------------------------
namespace utils {
//---------------------------------------------------------------------------
void printFusionYAML(const std::vector<bfuse::FusionInfo>& Infos)
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
void printKernelYAML(const std::map<std::string, bfuse::KernelInfo>& Infos)
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
} // namespace utils
//---------------------------------------------------------------------------