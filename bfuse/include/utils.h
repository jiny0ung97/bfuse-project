
#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include <map>

#include "llvm/Support/YAMLTraits.h"
#include "bfuse.h"
//---------------------------------------------------------------------------
namespace utils {
//---------------------------------------------------------------------------
template <typename Info>
Info readYAMLInfo(const std::string& Path)
{
  using FileOrError = llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>;

  FileOrError Buffer = llvm::MemoryBuffer::getFile(Path.c_str());
  if (!Buffer) {
    llvm::errs() << "BFUSE ERROR: failed to read configs\n";
    std::exit(0);
  }

  Info Infos;
  llvm::yaml::Input Yaml{Buffer.get()->getBuffer()};
  Yaml >> Infos;

  if (Yaml.error()) {
    llvm::errs() << "BFUSE ERROR: failed to get configs\n";
    std::exit(0);
  }

  return Infos;
}
//---------------------------------------------------------------------------
void printFusionYAML(const std::vector<bfuse::FusionInfo>& Infos);
//---------------------------------------------------------------------------
void printKernelYAML(const std::map<std::string, bfuse::KernelInfo>& Infos);
//---------------------------------------------------------------------------
} // namespace utils
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::KernelInfo> {
  static void mapping(llvm::yaml::IO &Io, bfuse::KernelInfo &Info)
  {
    Io.mapRequired("KernelName",  Info.kernelName);
    Io.mapRequired("HasBarriers", Info.hasBarriers);
    Io.mapRequired("GridDim",     Info.gridDim);
    Io.mapRequired("BlockDim",    Info.blockDim);
    Io.mapRequired("Reg",         Info.reg);

    Io.mapOptional("ExecTime", Info.execTime, 0.0);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::FusionInfo> {
  static void mapping(llvm::yaml::IO &Io, bfuse::FusionInfo &Info)
  {
    Io.mapRequired("File",    Info.filePath);
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