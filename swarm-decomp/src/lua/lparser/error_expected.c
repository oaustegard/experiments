#include "LexState.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaX_syntaxerror.h"
#include "fn_luaX_token2str.h"

static void error_expected (LexState *ls, int token) {
  luaX_syntaxerror(ls,
      luaO_pushfstring(ls->L, "%s expected", luaX_token2str(ls, token)));
}
