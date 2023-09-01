
#pragma once

#include <utility>
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
  inline std::tuple<int, const char **>getArguments() const { return std::make_tuple(argc, argv); }

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
struct KernelContext {
  using BlockIdxBoundaryVector = std::vector<std::pair<int, int>>;

  /// The kernel's information
  KernelInfo info;
  /// The kernel's threadIdx boudnary
  int threadBoundary;
  /// The kernel's blockIdx boundary
  BlockIdxBoundaryVector blockBoundaries;
  /// # of blocks from other fused kernels
  /// To rewrite blockIdx variables
  std::vector<int> otherBlocks;
};
//---------------------------------------------------------------------------
class FusionTool {
private:
  /// The order of fused kernels
  std::vector<std::string> kernels;
  /// The unordered map to contain kernel context
  std::unordered_map<std::string, KernelContext> kernelContextMap;

public:
  /// 
  constexpr static bool baseLine = false;
  ///
  constexpr static bool isBarSyncEnabled = false;
  ///
  constexpr static bool launchBound = false;
  ///
  constexpr static bool imBalancedThread = false;

  /// The constructor
  explicit FusionTool(std::vector<KernelInfo> Infos);
};
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------