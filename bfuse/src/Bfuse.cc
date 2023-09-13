
#include <cstdlib>
#include <string>

#include "bfuse/Contexts.h"
#include "bfuse/Tools.h"
#include "bfuse/Utils.h"
#include "bfuse/Bfuse.h"

using namespace std;
using namespace bfuse::tools;
using namespace bfuse::contexts;
//---------------------------------------------------------------------------
namespace bfuse {
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
void bfuse(const char *ProgName, string FusionInfoPath, string KernelInfoPath, string BasePath)
{
  // Extract information from yaml files
  auto FusionYAML = utils::readYAMLInfo<vector<FusionInfo>>(FusionInfoPath);
  auto KernelYAML = utils::readYAMLInfo<map<string, KernelInfo>>(KernelInfoPath);

  Arguments Arg{ProgName, BasePath};

  // Run block-level fusion
  for (auto& Info : FusionYAML) {
    FusionContext Context{Info, KernelYAML};
    FusionTool    Tool{Arg};

    // [Tests]
    // utils::printFusionContexts(Context);
    Tool.print(Context);
  }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------