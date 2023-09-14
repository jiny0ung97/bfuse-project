
#include <memory>
#include <vector>

#include "clang/Tooling/Tooling.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Contexts.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class FusionTool {
public:
  /// The AST of the source code
  std::vector<std::unique_ptr<clang::ASTUnit>> ASTs;
  
  /// The constructor
  explicit FusionTool(const CommonParsersArguments& Arg);
  /// print FusionTool
  void print() const;
  /// print FusionTool with given kernel name
  void print(std::string& KName) const;
};
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------