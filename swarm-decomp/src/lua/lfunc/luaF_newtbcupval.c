#include "fn_luaG_findlocal.h"
#include "fn_luaG_runerror.h"
#include "fn_luaT_gettmbyobj.h"
#include "fn_luaF_newtbcupval.h"

#define MAXDELTA ((1 << 16) - 1)
#define TBCDELTA(o) (*(unsigned short *)((char *)(o) + 10))

void luaF_newtbcupval (lua_State *L, StkId level) {
  if (l_isfalse(s2v(level)))
    return;
  {
    const TValue *tm = luaT_gettmbyobj(L, s2v(level), TM_CLOSE);
    if (ttisnil(tm)) {
      int idx = level - L->ci->func;
      const char *vname = luaG_findlocal(L, L->ci, idx, NULL);
      if (vname == NULL)
        vname = "?";
      luaG_runerror(L, "variable '%s' got a non-closable value", vname);
    }
  }
  while ((unsigned int)(level - L->tbclist) > MAXDELTA) {
    L->tbclist += MAXDELTA;
    TBCDELTA(L->tbclist) = 0;
  }
  TBCDELTA(level) = (unsigned short)(level - L->tbclist);
  L->tbclist = level;
}
