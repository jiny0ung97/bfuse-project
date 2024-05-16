
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <fstream>

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
void backUpFile(const string &FileName)
{
  llvm::sys::fs::copy_file(FileName, FileName + ".bak");
}
//---------------------------------------------------------------------------
void recoverFile(const string &FileName)
{
  llvm::sys::fs::copy_file(FileName + ".bak", FileName);
  llvm::sys::fs::remove(FileName + ".bak");
}
//---------------------------------------------------------------------------
void writeFile(const string &Path, const string &FileName, const string &Str)
{
  std::error_code FileErr;
  llvm::raw_fd_ostream Os(Path + "/" + FileName, FileErr, llvm::sys::fs::OF_None);

  Os << Str;
}
//---------------------------------------------------------------------------
} // namespace utils
} // namespace fuse
//---------------------------------------------------------------------------