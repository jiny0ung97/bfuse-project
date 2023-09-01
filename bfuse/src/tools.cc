
#include <utility>
#include <string>
#include <vector>
#include <map>
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
FusionTools FusionTools::create(FusionInfo& FInfo, map<string, KernelInfo>& KInfo)
{
  vector<KernelInfo> KInfoVector;
  for (auto& KName : FInfo.kernels) {
    KInfoVector.push_back(KInfo[KName]);
  }
  FusionTools tools(KInfoVector);

  return tools;
}
//---------------------------------------------------------------------------
FusionTools::FusionTools(vector<KernelInfo>& Infos)
{
  unordered_map<string, int> CurBounds;
  unordered_map<string, int> EndBounds;

  for (auto& info : Infos) {
    kernels.push_back(info.kernelName);

    CurBounds[info.kernelName] = 0;
    EndBounds[info.kernelName] = info.gridDim.size();
  }

  unordered_map<string, vector<pair<int, int>>> BlockBoundaries;
  unordered_map<string, vector<int>> OtherBlocks;
  const int TotalSM  = 84;
  int Idx            = 0;
  int TotalBounds    = 0;
  bool LastLoop      = false;

  while(true) {
    auto& KName = kernels[Idx];
    int Stride  = EndBounds[KName] - CurBounds[KName];

    if (!LastLoop && Stride > TotalSM)
      Stride = TotalSM;
    
    BlockBoundaries[KName].emplace_back(TotalBounds, TotalBounds + Stride);
    OtherBlocks[KName].push_back(TotalBounds - CurBounds[KName]);

    if (LastLoop)
      break;

    CurBounds[KName] += Stride;
    TotalBounds      += Stride;

    if (CurBounds[KName] == EndBounds[KName])
      LastLoop = true;

    Idx = (Idx + 1) % kernels.size();
  }

  for (auto& info : Infos) {
    auto& KName = info.kernelName;

    // TODO: need to summarize using constructor...
    kernelContextMap[KName].info            = info;
    kernelContextMap[KName].threadBoundary  = info.blockDim.size();
    kernelContextMap[KName].blockBoundaries = BlockBoundaries[KName];
    kernelContextMap[KName].otherBlocks     = OtherBlocks[KName];
  }
}
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------