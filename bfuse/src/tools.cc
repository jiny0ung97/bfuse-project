
#include <string>

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
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------