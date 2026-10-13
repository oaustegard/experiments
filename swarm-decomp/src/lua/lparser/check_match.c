#include "LexState.h"
#include "fn_luaX_syntaxerror.h"
#include "fn_luaX_token2str.h"
#include "fn_luaO_pushfstring.h"
#include "error_expected.h"
#include "testnext.h"
#include "check_match.h"
static void check_match (LexState *ls, int what, int who, int where) {
  if (__builtin_expect(!testnext(ls, what), 0)) {
    if (where == ls->linenumber)
      error_expected(ls, what);
    else
      luaX_syntaxerror(ls, luaO_pushfstring(ls->L,
                       "%s expected (to close %s at line %d)",
                       luaX_token2str(ls, what), luaX_token2str(ls, who), where));
  }
}
