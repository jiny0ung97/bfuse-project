
#pragma once

#include <string>
#include <tuple>

namespace fuse {
//---------------------------------------------------------------------------
class Arguments {
private:
  /// The argc parameter
  int Argc_;
  /// The argv parameter
  const char **Argv_;
  /// The path of compile_commands.json
  std::string CompileCommands_;
  /// The file path of fused kernels
  std::string File_;

public:
  /// Get argc, argv parameter
  std::tuple<int, const char **>getArguments() const { return std::make_tuple(Argc_, Argv_); }
  /// Print Arguments
  void print() const;

  /// The constructor
  Arguments(const std::string &ProgName, const std::string &CompileCommandsPath, const std::string &FilePath);
  /// The destructor
  ~Arguments();

  /// Delete default constructor
  Arguments() = delete;
  /// Delete copy constructor
  Arguments(const Arguments &other) = delete;
  /// Delete move constructor
  Arguments(Arguments &&other) = delete;
  /// Delete copy assignment operator
  Arguments& operator=(const Arguments &other) = delete;
  /// Delete move assignment operator
  Arguments& operator=(Arguments &&other) = delete;
};
//---------------------------------------------------------------------------
void bfuse(const std::string ProgName, const std::string FusionConfigPath,
           const std::string KernelConfigPath, const std::string CompileCommandsPath, const std::string OutputPath);
//---------------------------------------------------------------------------
void hfuse(const std::string ProgName, const std::string FusionConfigPath,
           const std::string KernelConfigPath, const std::string CompileCommandsPath, const std::string OutputPath);
//---------------------------------------------------------------------------
} // namespace fuse
//---------------------------------------------------------------------------