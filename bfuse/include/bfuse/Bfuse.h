
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
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
struct KernelInfo {
  /// The kernels' file path
  std::string filePath;
  /// Whether the kernel code has synchronization barriers
  bool hasBarriers;
  /// The kernel's grid dimension
  GridDim gridDim;
  /// The kernel's block dimension
  BlockDim blockDim;
};
//---------------------------------------------------------------------------
struct FusionInfo {
  /// The kernels to be fused
  std::vector<std::string> kernels;
};
//---------------------------------------------------------------------------
void bfuse(const char *ProgName, std::string FusionInfoPath,
           std::string KernelInfoPath, std::string BasePath);
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------