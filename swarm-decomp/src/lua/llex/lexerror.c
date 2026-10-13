#include "lexerror.h"
#include "txtToken.h"
#include "fn_luaG_addinfo.h"
#include "fn_luaD_throw.h"
#include "fn_luaO_pushfstring.h"

static _Noreturn void lexerror(LexState *ls, const char *msg, int token) {
  msg = luaG_addinfo(ls->L, msg, ls->source, ls->linenumber);
  if (token)
    luaO_pushfstring(ls->L, "%s near %s", msg, txtToken(ls, token));
  luaD_throw(ls->L, 3);
}
