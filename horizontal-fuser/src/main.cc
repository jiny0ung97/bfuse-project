
#include <cstdlib>
#include <iostream>
#include <string>
#include <getopt.h>

#include "fuse/Fuse.h"

using namespace std;
//---------------------------------------------------------------------------
static string ProgName            = "horizontal-fuser";
static string OutputFolder        = "./output";
static string FusionConfigPath    = "./fusions.json";
static string KernelConfigPath    = "./kernels.json";
static string CompileCommandsPath = "./compile_commands.json";
//---------------------------------------------------------------------------
static void print_help(const char *prog_name)
{
  cout << "Usage: " << prog_name << " [-h] [-o folder] fusion_config kernel_config compile_commands\n";
  cout << "Options:\n";
  cout << " -h                : Print this page.\n";
  cout << " -o                : Write output to the folder.           (default: ./output)\n";
  cout << "\n";
  cout << "fusion_config      : YAML file about fusion configuration.\n";
  cout << "kernel_config      : YAML file about kernel configuration.\n";
  cout << "compile_commands   : the \'compile_commands.json\' file.\n";
}
//---------------------------------------------------------------------------
static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "ho:")) != -1)
  {
    switch (c)
    {
    case 'o':
      OutputFolder = string(optarg);
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

  fuse::bfuse(ProgName, OutputFolder, FusionConfigPath, KernelConfigPath, CompileCommandsPath);

  return 0;
}