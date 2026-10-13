#include "llex_macros.h"
#include "lctype.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaX_token2str.h"

const char *luaX_token2str(LexState *ls, int token) {
  if (token <= 0xff) {
    if (lisprint(token))
      return luaO_pushfstring(ls->L, "'%c'", token);
    else
      return luaO_pushfstring(ls->L, "'<\\%d>'", token);
  } else {
    const char *s = luaX_tokens[token - 0x100];
    if (token <= 0x11f)
      return luaO_pushfstring(ls->L, "'%s'", s);
    else
      return s;
  }
}
