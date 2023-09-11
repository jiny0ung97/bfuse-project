
#pragma once

#include <string>
#include <vector>
#include <map>

#include "bfuse/Bfuse.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class FusionTools {
private:
  /// The kernels to be fused
  std::vector<std::string> kernels;
  /// The kernel's information
  std::map<std::string, KernelInfo> kernelInfoMap;
  /// The vector to contain kernel contexts
  std::map<std::string, KernelContext> kernelContextMap;

public:
  /// Get kernels to be fused
  std::vector<std::string> getKernelNames() const;
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