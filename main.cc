
#include <cstdlib>
#include <iostream>
#include <string>
#include <getopt.h>

#include "bfuse/BFuse.h"
#include "hfuse/HFuse.h"

using namespace std;
//---------------------------------------------------------------------------
static string CompileCommandsPath = ".";
static string ResultPath          = "./outputs/";
static string FuseConfigPath      = "./fusion.yaml";
static string KernelConfigPath    = "./kernel.yaml";
//---------------------------------------------------------------------------
static void print_help(const char *prog_name)
{
  cout << "usage: " << prog_name << " [<Options>...] FuseConfig KernelConfig\n\n";
  cout << "Options:\n";
  cout << "     -h                 : Print this page.\n";
  cout << "     -p <file>          : Specify \'compile_commands.json\' file path (default: .)\n";
  cout << "     -o <folder>        : Write outputs to folder                     (default: ./outputs/)\n";
  cout << "\n";
  cout << "FuseConfig              : YAML configuration file about block-/thread-level fusion\n";
  cout << "KernelConfig            : YAML configuration file about kernels to be fused\n";
}
//---------------------------------------------------------------------------
static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "hp:o:")) != -1)
  {
    switch (c)
    {
    case 'p':
      CompileCommandsPath = string(optarg);
      break;
    case 'o':
      ResultPath = string(optarg);
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
      FuseConfigPath = atoi(argv[i]);
      break;
    case 1:
      KernelConfigPath = atoi(argv[i]);
      break;
    default:
      break;
    }
  }

  cout << "================ Bfuse Plus Info ================\n";
  cout << "- The Path of Compile Commands:  " << CompileCommandsPath << "\n";
  cout << "- The Directory of Config files: " << ConfigFilePath << "\n";
  cout << "- The Directory of Result files: " << ResultPath << "\n\n";
}
//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  if (argc < 4) {
    print_help(argv[0]);
    exit(0);
  }
  
  ProgName = string(argv[0]);
  parse_opt(argc, argv);

#ifdef USE_BFUSE
  bfuse::bfuse(ProgName, CompileCommandsPath, ConfigFilePath, ResultPath);
#endif

  return 0;
}