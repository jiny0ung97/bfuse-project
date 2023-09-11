
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

#include "llvm/Support/YAMLTraits.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Tools.h"

// TODO: need to be changed into cxx style.
#define CHECK_ERROR(m)             \
  do                               \
  {                                \
    std::cerr << "bfuse ERROR ("   \
              << __FILE__ << ":"   \
              << __LINE__ << "): " \
              << m << "\n";        \
  } while (0)
//---------------------------------------------------------------------------
namespace bfuse {
namespace utils {
//---------------------------------------------------------------------------
template <typename Info>
Info readYAMLInfo(const std::string& Path)
{
  using FileOrError = llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>;

  FileOrError Buffer = llvm::MemoryBuffer::getFile(Path.c_str());
  if (!Buffer) {
    llvm::errs() << "[bfuse ERROR]: failed to read configs\n";
    std::exit(0);
  }

  Info Infos;
  llvm::yaml::Input Yaml{Buffer.get()->getBuffer()};
  Yaml >> Infos;

  if (Yaml.error()) {
    llvm::errs() << "[bfuse ERROR]: failed to get configs\n";
    std::exit(0);
  }

  return Infos;
}
//---------------------------------------------------------------------------
void printFusionInfo(const FusionInfo& Info);
//---------------------------------------------------------------------------
void printKernelInfo(const std::string& KName, const KernelInfo& Info);
//---------------------------------------------------------------------------
void printKernelContexts(const std::string& KName, const KernelContext& Context);
//---------------------------------------------------------------------------
void printFusionTools(const tools::FusionTools& Tools);
//---------------------------------------------------------------------------
} // namespace utils
} // namespace bfuse
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::KernelInfo> {
  static void mapping(llvm::yaml::IO &Io, bfuse::KernelInfo &Info)
  {
    Io.mapRequired("HasBarriers", Info.hasBarriers);
    Io.mapRequired("GridDim",     Info.gridDim);
    Io.mapRequired("BlockDim",    Info.blockDim);
    Io.mapRequired("File",        Info.filePath);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::FusionInfo> {
  static void mapping(llvm::yaml::IO &Io, bfuse::FusionInfo &Info)
  {
    Io.mapRequired("Kernels", Info.kernels);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::GridDim> {
  static void mapping(llvm::yaml::IO &Io, bfuse::GridDim &Dim)
  {
    Io.mapRequired("X", Dim.x);
    Io.mapRequired("Y", Dim.y);
    Io.mapRequired("Z", Dim.z);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::BlockDim> {
  static void mapping(llvm::yaml::IO &Io, bfuse::BlockDim &Dim)
  {
    Io.mapRequired("X", Dim.x);
    Io.mapRequired("Y", Dim.y);
    Io.mapRequired("Z", Dim.z);
  }
};
//---------------------------------------------------------------------------
LLVM_YAML_IS_SEQUENCE_VECTOR(bfuse::FusionInfo)
LLVM_YAML_IS_STRING_MAP(bfuse::KernelInfo)
//---------------------------------------------------------------------------