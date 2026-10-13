#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "Labeldesc.h"
#include "getlocalvardesc.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaK_semerror.h"

static void jumpscopeerror (LexState *ls, Labeldesc *gt) {
  const char *varname = getlocalvardesc(ls->fs, gt->nactvar)->vd.name->contents;
  const char *msg = "<goto %s> at line %d jumps into the scope of local '%s'";
  msg = luaO_pushfstring(ls->L, msg, gt->name->contents, gt->line, varname);
  luaK_semerror(ls, msg);
}
