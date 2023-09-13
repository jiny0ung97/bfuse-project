
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
private:
  /// The AST of the source code
  std::vector<std::unique_ptr<clang::ASTUnit>> aSTs;

public:
  /// print FusionTool
  void print(contexts::FusionContext& Context) const;
  
  /// The constructor
  explicit FusionTool(const Arguments& Arg);
};
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------