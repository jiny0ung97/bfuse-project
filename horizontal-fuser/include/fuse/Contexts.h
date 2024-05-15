
#pragma once

#include <utility>
#include <string>
#include <vector>
#include <map>
//---------------------------------------------------------------------------
namespace fuse {
//---------------------------------------------------------------------------
namespace contexts {
//---------------------------------------------------------------------------
struct GridDim {
  /// Grid's x, y, z dimension
  int X, Y, Z;
  /// The size of grid dimension
  int size() const { return X * Y * Z; }

};
//---------------------------------------------------------------------------
struct BlockDim {
  /// Block's x, y, z dimension
  int X, Y, Z;
  /// The size of block dimension
  int size() const { return X * Y * Z; }
};
//---------------------------------------------------------------------------
class KernelInfo {
public:
  /// The kernel's name
  std::string KernelName_;
  /// Whether the kernel code has synchronization barriers
  bool HasBarriers_;
  /// The kernel's grid dimension
  GridDim GridDim_;
  /// The kernel's block dimension
  BlockDim BlockDim_;
  /// The count of kernel's used register
  int Reg_;
  /// The kernel's execution time
  float ExecTime_;

  /// Print KernelInfo
  void print(const std::string &KName) const;
};
//---------------------------------------------------------------------------
class FusionInfo {
public:
  /// The kernels' file path
  std::string File_;
  /// The kernels to be fused
  std::vector<std::string> Kernels_;

  /// Print FusionInfo
  void print() const;
};
//---------------------------------------------------------------------------
class FusionContext {
public:
  /// The kernels to be fused
  std::vector<std::string> Kernels_;
  /// The kernel's information
  std::map<std::string, KernelInfo> KernelInfoMap_;

  /// The constructor
  // FusionContext(FusionInfo &FInfo, std::map<std::string, KernelInfo> &KInfoMap);
  /// The constructor
  FusionContext(std::vector<std::string> &&Kernels, std::map<std::string, KernelInfo> &&KernelInfoMap)
                : Kernels_{std::move(Kernels)}, KernelInfoMap_{std::move(KernelInfoMap)} {}
  /// Print FusionContext
  void print() const;

  /// Create the KernelContext
  static FusionContext create(FusionInfo &FInfo, std::map<std::string, KernelInfo> &KInfoMap);
};
//---------------------------------------------------------------------------
} // namespace contexts
} // namespace fuse
//---------------------------------------------------------------------------