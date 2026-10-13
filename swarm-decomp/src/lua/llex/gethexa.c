#include "llex_macros.h"
#include "lctype.h"
#include "save.h"
#include "esccheck.h"
#include "fn_luaO_hexavalue.h"

static int gethexa(LexState *ls) {
  save(ls, ls->current);
  next(ls);
  esccheck(ls, lisxdigit(ls->current), "hexadecimal digit expected");
  return luaO_hexavalue(ls->current);
}
