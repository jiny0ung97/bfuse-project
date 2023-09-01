
#include <utility>
#include <string>
#include <vector>
#include <unordered_map>

#include "bfuse.h"
#include "tools.h"

using namespace std;
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
Arguments::Arguments(const char *ProgName, string& Path)
{
  filePath = Path;

  argv    = (const char**)malloc(sizeof(char *) * 2);
  argv[0] = ProgName;
  argv[1] = filePath.c_str();
}
//---------------------------------------------------------------------------
Arguments::~Arguments() { free(argv); }
//---------------------------------------------------------------------------
FusionTool::FusionTool(const vector<KernelInfo> Infos)
{
  for (auto& info : Infos) {
    kernels.push_back(info.kernelName);
  }

  unordered_map<string, pair<int, int>> BlockBoundary;
  unordered_map<string, vector<int>>    BlockLeft;
  int Idx      = 0;
  int CurBound = 0;
  constexpr int TotalSM = 84;

  while(true) {
    auto& KName = kernels[Idx];
    // int Stride = Bou
  }
}
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------