
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>

#include "bfuse.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class Arguments {
private:
  /// The argc parameter
  int argc = 2;
  /// The argv parameter
  const char **argv;
  /// The file path of fused kernels
  std::string filePath;

public:
  /// Get argc, argv parameter
  std::tuple<int, const char **>getArguments() const { return std::make_tuple(argc, argv); }

  /// The constructor
  Arguments(const char *ProgName, std::string& Path);
  /// The destructor
  ~Arguments();

  /// Delete default constructor
  Arguments() = delete;
  /// Delete copy constructor
  Arguments(const Arguments& other) = delete;
  /// Delete move constructor
  Arguments(Arguments&& other) = delete;
  /// Delete copy assignment operator
  Arguments& operator=(const Arguments& other) = delete;
  /// Delete move assignment operator
  Arguments& operator=(Arguments&& other) = delete;
};
//---------------------------------------------------------------------------
class Parser {
private:
  ///
  std::unordered_map<std::string, KernelInfo> kernelMap;
  ///
  std::unordered_map<std::string, std::pair<unsigned, unsigned>> bounds;

public:
  /// The constructor
  Parser(int argc, const char** argv, const std::vector<KernelInfo>& Infos);
};
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------