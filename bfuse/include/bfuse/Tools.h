
#pragma once

#include <utility>
#include <string>
#include <vector>
#include <map>
#include <tuple>

#include "bfuse/Bfuse.h"
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
struct KernelContext {
  using IdxBoundPair = std::pair<int, int>;

  /// The kernel's threadIdx boudnary
  IdxBoundPair threadIdxInfo;
  /// The kernel's blockIdx boundary
  std::vector<IdxBoundPair> blockIdxInfo;
  /// # of blocks from other fused kernels
  /// To rewrite blockIdx variables
  std::vector<int> otherBlocks;
};
//---------------------------------------------------------------------------
class FusionTools {
private:
  /// The kernel's information
  std::map<std::string, KernelInfo> kernelInfoMap;
  /// The vector to contain kernel contexts
  std::map<std::string, KernelContext> kernelContextMap;

public:
  /// Get KernelInfo of given kernel name
  KernelInfo getKernelInfo(const std::string& KName) const;
  /// Get KernelContext of given kernel name
  KernelContext getKernelContext(const std::string& KName) const;

  /// The constructor
  FusionTools(FusionInfo& FInfo, std::map<std::string, KernelInfo>& KInfo);
};
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------