#include "llex_macros.h"
#include "fn_luaZ_fill.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaM_saferealloc_.h"
#include "fn_luaX_setinput.h"

void luaX_setinput(lua_State *L, LexState *ls, ZIO *z, TString *source, int firstchar) {
  ls->t.token = 0;
  ls->L = L;
  ls->current = firstchar;
  ls->lookahead.token = 0x120;
  ls->z = z;
  ls->fs = NULL;
  ls->linenumber = 1;
  ls->lastline = 1;
  ls->source = source;
  ls->envn = luaS_newlstr(L, "_ENV", sizeof("_ENV") - 1);
  ls->buff->buffer = (char *)luaM_saferealloc_(ls->L, ls->buff->buffer, ls->buff->buffsize, 32);
  ls->buff->buffsize = 32;
}
