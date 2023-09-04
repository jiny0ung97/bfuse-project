
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
  return FusionTools{KInfoVector};
}
//---------------------------------------------------------------------------
FusionTools::FusionTools(vector<KernelInfo>& Infos)
{
  unordered_map<string, int> CurBounds;
  unordered_map<string, int> EndBounds;

  unordered_map<string, IdxBoundPair>         ThreadIdxInfo;
  unordered_map<string, vector<IdxBoundPair>> BlockIdxInfo;
  unordered_map<string, vector<int>>          OtherBlocks;

  const int TotalSM  = 84;
  int Idx            = 0;
  int TotalBounds    = 0;
  bool LastLoop      = false;

  for (auto& info : Infos) {
    auto& KName = info.kernelName;

    CurBounds[KName]     = 0;
    EndBounds[KName]     = info.gridDim.size();
    ThreadIdxInfo[KName] = make_pair(0, info.blockDim.size());
  }

  while(true) {
    auto& KName = Infos[Idx].kernelName;
    int Stride  = EndBounds[KName] - CurBounds[KName];

    if (!LastLoop && Stride > TotalSM)
      Stride = TotalSM;
    
    BlockIdxInfo[KName].emplace_back(TotalBounds, TotalBounds + Stride);
    OtherBlocks[KName].push_back(TotalBounds - CurBounds[KName]);

    if (LastLoop)
      break;

    CurBounds[KName] += Stride;
    TotalBounds      += Stride;

    if (CurBounds[KName] == EndBounds[KName])
      LastLoop = true;

    Idx = (Idx + 1) % Infos.size();
  }

  for (auto& info : Infos) {
    auto& KName = info.kernelName;
    kernelContexts.emplace_back(info, ThreadIdxInfo[KName],
                                BlockIdxInfo[KName], OtherBlocks[KName]);
  }
}
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------