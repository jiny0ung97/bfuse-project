
#include <cstdlib>
#include <iostream>
#include <string>
#include <getopt.h>

#ifdef USE_BFUSE
#include "bfuse/Bfuse.h"
#endif

using namespace std;
//---------------------------------------------------------------------------
static string ProgName            = "/usr/local/bin/clang++";
static string CompileCommandsPath = ".";
static string ConfigFilePath      = "config";
static string ResultPath          = ".";
//---------------------------------------------------------------------------
static void print_help(const char *prog_name)
{
  cout << "Usage: " << prog_name << " [-h] [-p compile_commands_path] [-c config_file_path] [-d result_path]\n";
  cout << "Options:\n";
  cout << "     -h : print this page.\n";
  cout << "     -p : the path of \'compile_commands.json\' file (default: .)\n";
  cout << "     -c : the directory of configuration files     (default: ./config)\n";
  cout << "     -d : The directory of result files            (default: .)\n\n";
}
//---------------------------------------------------------------------------
static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "hp:c:d:")) != -1)
  {
    switch (c)
    {
    case 'p':
      CompileCommandsPath = string(optarg);
      break;
    case 'c':
      ConfigFilePath = string(optarg);
      break;
    case 'd':
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