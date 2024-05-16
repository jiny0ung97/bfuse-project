
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
  /// The number of GPU's SM
  int TotalSM_;
  /// The kernels to be fused
  std::vector<std::string> Kernels_;
  /// The kernel's information
  std::map<std::string, KernelInfo> KernelInfoMap_;
  /// The fused kernel's name
  std::string FusedKernelName_;
  /// The fused kernel's GridDim;
  GridDim FusedGridDim_;
  /// The fused kernel's BlockDim;
  BlockDim FusedBlockDim_;
  /// The fused kernel's new cuda built-in declarations
  std::string FusedBlockDeclStr_;
  /// The fused kernel's branch condition for each kernel
  std::map<std::string, std::string> FusedCondStrMap_;

  /// The constructor
  // FusionContext(FusionInfo &FInfo, std::map<std::string, KernelInfo> &KInfoMap);
  /// The constructor
  FusionContext(int TotalSM, std::vector<std::string> &&Kernels, std::map<std::string, KernelInfo> &&KernelInfoMap,
                std::string &&FusedKernelName, GridDim &&FusedGridDim, BlockDim &&FusedBlockDim,
                std::string &&FusedBlockDeclStr, std::map<std::string, std::string> &&FusedCondStrMap)
                : TotalSM_{TotalSM}, Kernels_{std::move(Kernels)}, KernelInfoMap_{std::move(KernelInfoMap)},
                  FusedKernelName_{std::move(FusedKernelName)}, FusedGridDim_{std::move(FusedGridDim)}, FusedBlockDim_{std::move(FusedBlockDim)},
                  FusedBlockDeclStr_{std::move(FusedBlockDeclStr)}, FusedCondStrMap_{std::move(FusedCondStrMap)} {}
  /// Print FusionContext
  void print() const;

  /// Create the KernelContext
  static FusionContext create(FusionInfo &FInfo, std::map<std::string, KernelInfo> &KInfoMap);
};
//---------------------------------------------------------------------------
} // namespace contexts
} // namespace fuse
//---------------------------------------------------------------------------