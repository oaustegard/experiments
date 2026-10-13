#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "Labeldesc.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaK_semerror.h"
#include "findlabel.h"

static void checkrepeated (LexState *ls, TString *name) {
  Labeldesc *lb = findlabel(ls, name);
  if (lb != NULL) {
    const char *msg = "label '%s' already defined on line %d";
    msg = luaO_pushfstring(ls->L, msg, name->contents, lb->line);
    luaK_semerror(ls, msg);
  }
}
