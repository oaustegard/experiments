#include "LexState.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaK_semerror.h"
#include "checknext.h"
#include "str_checkname.h"
#include "fn_luaX_next.h"
#include <string.h>

static int getlocalattribute (LexState *ls) {
  if (ls->t.token == '<') {
    const char *attr;
    luaX_next(ls);
    attr = str_checkname(ls)->contents;
    checknext(ls, '>');
    if (strcmp(attr, "const") == 0)
      return 1;
    else if (strcmp(attr, "close") == 0)
      return 2;
    else
      luaK_semerror(ls, luaO_pushfstring(ls->L, "unknown attribute '%s'", attr));
  }
  return 0;
}
