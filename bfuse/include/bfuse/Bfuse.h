
#pragma once

#include <string>
#include <tuple>

namespace bfuse {
//---------------------------------------------------------------------------
class OptionsParserArguments {
private:
  /// The argc parameter
  int argc = 4;
  /// The argv parameter
  const char **argv;
  /// The path of compile_commands.json
  std::string compileCommandsPath;
  /// The file path of fused kernels
  std::string filePath;

public:
  /// Get argc, argv parameter
  std::tuple<int, const char **>getArguments() const { return std::make_tuple(argc, argv); }

  /// The constructor
  OptionsParserArguments(const char *ProgName, std::string& CompileCommandsPath, std::string& FilePath);
  /// The destructor
  ~OptionsParserArguments();

  /// Delete default constructor
  OptionsParserArguments() = delete;
  /// Delete copy constructor
  OptionsParserArguments(const OptionsParserArguments& other) = delete;
  /// Delete move constructor
  OptionsParserArguments(OptionsParserArguments&& other) = delete;
  /// Delete copy assignment operator
  OptionsParserArguments& operator=(const OptionsParserArguments& other) = delete;
  /// Delete move assignment operator
  OptionsParserArguments& operator=(OptionsParserArguments&& other) = delete;
};
//---------------------------------------------------------------------------
void bfuse(const char *ProgName, std::string ConfigFilePath, std::string CompileCommandsPath);
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------