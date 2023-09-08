
#include <utility>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "bfuse/Bfuse.h"
#include "bfuse/Utils.h"
#include "bfuse/Tools.h"
// #include "bfuse/MatchFinders.h"

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
FusionTools::FusionTools(FusionInfo& FInfo, map<string, KernelInfo>& KInfo)
{
  vector<string> Order;
  map<string, int> CurBounds;
  map<string, int> EndBounds;
  map<string, KernelContext> KernelContextMap;

  const int TotalSM  = 84;
  int Idx            = 0;
  int TotalBounds    = 0;
  bool LastLoop      = false;

  for (auto& KName : FInfo.kernels) {
    auto& Info = KInfo.find(KName)->second;

    Order.push_back(KName);
    CurBounds[KName]     = 0;
    EndBounds[KName]     = Info.gridDim.size();
    kernelInfoMap[KName] = Info;

    KernelContext Context;
    Context.threadIdxInfo   = make_pair(0, Info.blockDim.size());
    KernelContextMap[KName] = move(Context);
  }

  while(true) {
    auto& KName         = Order[Idx];
    auto& Context       = KernelContextMap.find(KName)->second;
    auto& BlockIdxInfo  = Context.blockIdxInfo;
    auto& OtherBlocks   = Context.otherBlocks;

    int Stride  = EndBounds[KName] - CurBounds[KName];

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

    Idx = (Idx + 1) % Order.size();
  }

  kernelContextMap = move(KernelContextMap);
}
//---------------------------------------------------------------------------
KernelInfo FusionTools::getKernelInfo(const string& KName) const
{
  auto KernelInfoMapIter = kernelInfoMap.find(KName);
  if (KernelInfoMapIter == kernelInfoMap.end()) {
    CHECK_ERROR("Cannot find KernelInfo with given name.");
    exit(0);
  }
  return KernelInfoMapIter->second;
}
//---------------------------------------------------------------------------
KernelContext FusionTools::getKernelContext(const string& KName) const
{
  auto KernelContextMapIter = kernelContextMap.find(KName);
  if (KernelContextMapIter == kernelContextMap.end()) {
    CHECK_ERROR("Cannot find KernelContext with given name.");
    exit(0);
  }
  return KernelContextMapIter->second;
}
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------