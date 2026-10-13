#include <stddef.h>
#include "lstate.h"
#include "Udata.h"
#include "Udata0.h"
#include "fn_luaM_toobig.h"
#include "fn_luaC_newobj.h"
#include "fn_luaS_newudata.h"

#define l_unlikely(x) __builtin_expect(((x) != 0), 0)
#define MAX_SIZE 0x7fffffffffffffffULL
#define udatamemoffset(nuv) \
  ((nuv) == 0 ? offsetof(Udata0, bindata) \
              : offsetof(Udata, uv) + (sizeof(UValue) * (nuv)))

Udata *luaS_newudata (lua_State *L, size_t s, int nuvalue) {
  Udata *u;
  int i;
  GCObject *o;
  if (l_unlikely(s > MAX_SIZE - udatamemoffset(nuvalue)))
    luaM_toobig(L);
  o = luaC_newobj(L, 7, udatamemoffset(nuvalue) + s);
  u = (Udata *)o;
  u->len = s;
  u->nuvalue = nuvalue;
  u->metatable = NULL;
  for (i = 0; i < nuvalue; i++)
    u->uv[i].uv.tt_ = 0;
  return u;
}
