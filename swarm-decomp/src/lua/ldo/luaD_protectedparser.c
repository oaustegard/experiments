#include "SParser.h"
#include "fn_luaD_pcall.h"
#include "fn_luaM_free_.h"
#include "fn_luaM_saferealloc_.h"
#include "Pfunc.h"

void f_parser (lua_State *L, void *ud);

int luaD_protectedparser (lua_State *L, ZIO *z, const char *name, const char *mode) {
  struct SParser p;
  int status;
  L->nCcalls += 0x10000;
  p.z = z; p.name = name; p.mode = mode;
  p.dyd.actvar.arr = NULL; p.dyd.actvar.size = 0;
  p.dyd.gt.arr = NULL; p.dyd.gt.size = 0;
  p.dyd.label.arr = NULL; p.dyd.label.size = 0;
  p.buff.buffer = NULL; p.buff.buffsize = 0;
  status = luaD_pcall(L, f_parser, &p, (char *)L->top - (char *)L->stack, L->errfunc);
  p.buff.buffer = luaM_saferealloc_(L, p.buff.buffer, p.buff.buffsize, 0);
  p.buff.buffsize = 0;
  luaM_free_(L, p.dyd.actvar.arr, p.dyd.actvar.size * sizeof(Vardesc));
  luaM_free_(L, p.dyd.gt.arr, p.dyd.gt.size * sizeof(Labeldesc));
  luaM_free_(L, p.dyd.label.arr, p.dyd.label.size * sizeof(Labeldesc));
  L->nCcalls -= 0x10000;
  return status;
}
