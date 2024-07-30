
#include <cstdlib>
#include <iostream>
#include <string>
#include <getopt.h>

#include "fuse/Fuse.h"

using namespace std;
//---------------------------------------------------------------------------
static string ProgName            = "horizontal-fuser";
static string OutputPath          = ".";
static string YAMLOutputPath      = ".";
static string FusionConfigPath    = "fusions.json";
static string KernelConfigPath    = "kernels.json";
static string CompileCommandsPath = ".";

static bool ExecBFuse = false;
//---------------------------------------------------------------------------
static void print_help(const char *prog_name)
{
  cout << "Usage: " << prog_name << " [-h] [-b] [-y yaml_output] [-o output] fusion_config kernel_config compile_commands\n";
  cout << "Options:\n";
  cout << " -h                : Print this page.\n";
  cout << " -b                : Process Block-level fusion.                   (default: false)\n";
  cout << " -y                : Write output yaml to the directory.           (default: .)\n";
  cout << " -o                : Write output file to the directory.           (default: .)\n";
  cout << "\n";
  cout << "fusion_config      : YAML file about fusion configuration.\n";
  cout << "kernel_config      : YAML file about kernel configuration.\n";
  cout << "compile_commands   : the directory where \'compile_commands.json\' file is located.\n";
}
//---------------------------------------------------------------------------
static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "hbo:y:")) != -1)
  {
    switch (c)
    {
    case 'o':
      OutputPath = string(optarg);
      break;
    case 'y':
      YAMLOutputPath = string(optarg);
      break;
    case 'b':
      ExecBFuse = true;
      break;
    case 'h':
    default:
      print_help(argv[0]);
      exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j)
  {
    switch (j)
    {
    case 0:
      FusionConfigPath = string(argv[i]);
      break;
    case 1:
      KernelConfigPath = string(argv[i]);
      break;
    case 2:
      CompileCommandsPath = string(argv[i]);
      break;
    default:
      break;
    }
  }

  cout << "fusion config: " << FusionConfigPath << "\n";
  cout << "kernel config: " << KernelConfigPath << "\n";
  cout << "compile commands: " << CompileCommandsPath << "\n";
}
//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  if (argc < 3) {
    print_help(argv[0]);
    exit(0);
  }
  
  ProgName = string(argv[0]);
  parse_opt(argc, argv);

  if (ExecBFuse) {
    fuse::bfuse(ProgName, FusionConfigPath, KernelConfigPath, CompileCommandsPath, OutputPath, YAMLOutputPath);
  } else {
    fuse::hfuse(ProgName, FusionConfigPath, KernelConfigPath, CompileCommandsPath, OutputPath, YAMLOutputPath);
  }

  return 0;
}
//---------------------------------------------------------------------------