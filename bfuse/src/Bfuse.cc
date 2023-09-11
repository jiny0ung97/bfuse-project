
#include <cstdlib>
#include <string>

#include "bfuse/Contexts.h"
#include "bfuse/Tools.h"
#include "bfuse/Utils.h"
#include "bfuse/Bfuse.h"

using namespace std;
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
  tools::FusionTool Tool{Arg};

  Tool.print();

  // // Run block-level fusion
  // for (auto& Info : FusionYAML) {
  //   // Create fusion tools object
  //   contexts::FusionContext Context{Info, KernelYAML};

  //   // [Tests]
  //   utils::printFusionContexts(Context);
  // }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------