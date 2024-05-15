
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

#include "llvm/Support/FileSystem.h"

#include "fuse/Utils.h"
#include "fuse/Contexts.h"

using namespace std;
//---------------------------------------------------------------------------
namespace fuse {
namespace utils {
//---------------------------------------------------------------------------
// Assume that fusion info has right definition
// run after checkFusionValid()
string extractFilePath(contexts::FusionInfo& FInfo)
{
  return FInfo.File_;
}
//---------------------------------------------------------------------------
void backUpFiles(const string &FileName)
{
  llvm::sys::fs::copy_file(FileName, FileName + ".bak");
}
//---------------------------------------------------------------------------
void recoverFiles(const string &FileName)
{
  llvm::sys::fs::copy_file(FileName + ".bak", FileName);
  llvm::sys::fs::remove(FileName + ".bak");
}
//---------------------------------------------------------------------------
} // namespace utils
} // namespace fuse
//---------------------------------------------------------------------------