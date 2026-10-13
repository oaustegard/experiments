#include "LexState.h"
#include "Labeldesc.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaK_semerror.h"

static void undefgoto (LexState *ls, Labeldesc *gt) {
  const char *msg;
  if (gt->name == luaS_newlstr(ls->L, "break", 5)) {
    msg = "break outside loop at line %d";
    msg = luaO_pushfstring(ls->L, msg, gt->line);
  }
  else {
    msg = "no visible label '%s' for <goto> at line %d";
    msg = luaO_pushfstring(ls->L, msg, gt->name->contents, gt->line);
  }
  luaK_semerror(ls, msg);
}
