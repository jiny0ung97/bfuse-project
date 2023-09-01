
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
  unordered_map<string, int> Bounds;
  for (auto& info : Infos) {
    kernels.push_back(info.kernelName);
    Bounds[info.kernelName] = info.gridDim.size();
  }

  unordered_map<string, vector<pair<int, int>>> BlockBoundaries;
  unordered_map<string, vector<int>>    Base;
  int Idx      = 0;
  int CurBound = 0;
  constexpr int TotalSM = 84;

  
  while(true) {
    auto& KName = kernels[Idx];
    int Stride  = Bounds[KName] > TotalSM ? TotalSM : Bounds[KName];

    BlockBoundaries[KName].emplace_back(CurBound, CurBound + Stride);
    CurBound += Stride;
    Bounds[KName] -= Stride;
    Idx = (Idx + 1) % kernels.size();
    Base[kernels[Idx]].push_back(Stride);
  }
}
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------