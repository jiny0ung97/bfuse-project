
#pragma once

#include <cstdlib>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include <tuple>
//---------------------------------------------------------------------------
namespace bfuse {
//---------------------------------------------------------------------------
struct GridDim {
  /// Grid's x, y, z dimension
  int x, y, z;
  /// The size of grid dimension
  int size() const { return x * y * z; }

};
//---------------------------------------------------------------------------
struct BlockDim {
  /// Block's x, y, z dimension
  int x, y, z;
  /// The size of block dimension
  int size() const { return x * y * z; }
};
//---------------------------------------------------------------------------
class KernelInfo {
public:
  /// The kernels' file path
  std::string filePath;
  /// Whether the kernel code has synchronization barriers
  bool hasBarriers;
  /// The kernel's grid dimension
  GridDim gridDim;
  /// The kernel's block dimension
  BlockDim blockDim;

  /// Print KernelInfo
  void print(const std::string& KName) const;
};
//---------------------------------------------------------------------------
class FusionInfo {
public:
  /// The kernels to be fused
  std::vector<std::string> kernels;

  /// Print FusionInfo
  void print() const;
};
//---------------------------------------------------------------------------
class CommonParsersArguments {
private:
  /// The argc parameter
  int argc = 4;
  /// The argv parameter
  const char **argv;
  /// The path of compile_commands.json
  std::string compileCommandsPath;
  /// The file path of fused kernels
  std::string filePath;

public:
  /// Get argc, argv parameter
  std::tuple<int, const char **>getArguments() const { return std::make_tuple(argc, argv); }

  /// The constructor
  CommonParsersArguments(const char *ProgName, std::string& CompileCommandsPath, std::string& FilePath);
  /// The destructor
  ~CommonParsersArguments();

  /// Delete default constructor
  CommonParsersArguments() = delete;
  /// Delete copy constructor
  CommonParsersArguments(const CommonParsersArguments& other) = delete;
  /// Delete move constructor
  CommonParsersArguments(CommonParsersArguments&& other) = delete;
  /// Delete copy assignment operator
  CommonParsersArguments& operator=(const CommonParsersArguments& other) = delete;
  /// Delete move assignment operator
  CommonParsersArguments& operator=(CommonParsersArguments&& other) = delete;
};
//---------------------------------------------------------------------------
void bfuse(const char *ProgName, std::string ConfigFilePath, std::string CompileCommandsPath);
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------