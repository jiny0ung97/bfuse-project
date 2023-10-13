
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

#include "llvm/Support/FileSystem.h"

#include "bfuse/Contexts.h"
#include "bfuse/Utils.h"

using namespace std;

using namespace bfuse::contexts;
//---------------------------------------------------------------------------
namespace bfuse {
namespace utils {
//---------------------------------------------------------------------------
// Assume that fusion info has right definition
// run after checkFusionValid()
string extractFilePath(FusionInfo& FInfo)
{
  return FInfo.filePath;
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
} // namespace bfuse
//---------------------------------------------------------------------------