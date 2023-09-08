
#include <cstdlib>
#include <iostream>
#include <string>
#include <getopt.h>

#ifdef USE_BFUSE
#include "bfuse/Bfuse.h"
#endif

using namespace std;
//---------------------------------------------------------------------------
static string FusionInfoPath;
static string KernelInfoPath;
static string BasePath;
//---------------------------------------------------------------------------
static void print_help(const char *prog_name)
{
  cout << "Usage: " << prog_name << " [-h] F K B\n";
  cout << "Options:\n";
  cout << "     -h : print this page.\n";
  cout << "      F : file path of fusion informations\n";
  cout << "      K : file path of kernel informations\n";
  cout << "      B : base path of generated kernels by Clang\n";
}
//---------------------------------------------------------------------------
static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "h")) != -1)
  {
    switch (c)
    {
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
        FusionInfoPath = argv[i];
        break;
      case 1:
        KernelInfoPath = argv[i];
        break;
      case 2:
        BasePath = argv[i];
        break;
      default:
        break;
    }
  }

  cout << "================ Bfuse Plus Info ================\n";
  cout << "- Path of Fusion Info: " << FusionInfoPath << "\n";
  cout << "- Path of Kernel Info: " << KernelInfoPath << "\n";
  cout << "- Base Path of Generated Kernels: " << BasePath << "\n";
}
//---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  if (argc < 4) {
    print_help(argv[0]);
    abort();
  }
  
  parse_opt(argc, argv);

#ifdef USE_BFUSE
  bfuse::bfuse(argv[0], FusionInfoPath, KernelInfoPath, BasePath);
#endif

  return 0;
}