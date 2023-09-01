
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
} // namespace utils
} // namespace bfuse
//---------------------------------------------------------------------------