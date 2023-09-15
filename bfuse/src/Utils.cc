
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

#include "bfuse/Contexts.h"
#include "bfuse/Utils.h"

using namespace std;

using namespace bfuse::contexts;
//---------------------------------------------------------------------------
namespace bfuse {
namespace utils {
//---------------------------------------------------------------------------
bool checkFusionValid(FusionInfo& FInfo, map<string, KernelInfo>& KInfoMap)
{
  vector<string> FileInfos;

  // check kernels to be fused exist
  for (auto& KName : FInfo.kernels) {
    auto KInfoMapIter = KInfoMap.find(KName);
    if (KInfoMapIter == KInfoMap.end())
      return false;

    auto& KInfo = KInfoMapIter->second;
    FileInfos.push_back(KInfo.filePath);
  }

  // check all kernels have different names
  auto& V = FInfo.kernels;
  V.erase(unique(V.begin(), V.end()), V.end());
  if (V.size() != FInfo.kernels.size())
    return false;

  // check all kernels exist in same file
  for (auto& FName : FileInfos) {
    if (FName != FileInfos.front())
      return false;
  }
  return true;
}
//---------------------------------------------------------------------------
// Assume that fusion info has right definition
// run after checkFusionValid()
string extractFilePath(FusionInfo& FInfo, map<string, KernelInfo>& KInfoMap)
{
  for (auto& KName : FInfo.kernels) {
    auto& KInfo = KInfoMap.find(KName)->second;
    return KInfo.filePath;
  }

  // Never reach here
  return "";
}
//---------------------------------------------------------------------------
} // namespace utils
} // namespace bfuse
//---------------------------------------------------------------------------