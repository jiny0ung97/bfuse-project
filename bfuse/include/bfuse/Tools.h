
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
  std::vector<std::unique_ptr<clang::ASTUnit>> aSTs;
  
  /// The constructor
  explicit FusionTool(const Arguments& Arg);
  /// print FusionTool
  void print() const;
};
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------