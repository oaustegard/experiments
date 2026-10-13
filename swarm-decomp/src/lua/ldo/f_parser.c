#include <string.h>
#include "SParser.h"
#include "fn_luaU_undump.h"
#include "fn_luaY_parser.h"
#include "fn_luaZ_fill.h"
#include "fn_luaF_initupvals.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaD_throw.h"

static void checkmode (lua_State *L, const char *mode, const char *x) {
  if (mode && !strchr(mode, x[0])) {
    luaO_pushfstring(L, "attempt to load a %s chunk (mode is '%s')", x, mode);
    luaD_throw(L, 3);
  }
}

static void f_parser (lua_State *L, void *ud) {
  LClosure *cl;
  struct SParser *p = (struct SParser *)ud;
  int c;
  {
    ZIO *z = p->z;
    if (z->n-- > 0) { c = (unsigned char)*z->p++; } else { c = luaZ_fill(z); }
  }
  if (c == 0x1b) {
    checkmode(L, p->mode, "binary");
    cl = luaU_undump(L, p->z, p->name);
  } else {
    checkmode(L, p->mode, "text");
    cl = luaY_parser(L, p->z, &p->buff, &p->dyd, p->name, c);
  }
  luaF_initupvals(L, cl);
}
