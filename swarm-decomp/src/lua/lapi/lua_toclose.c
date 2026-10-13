#include "lstate.h"
#include "fn_luaF_newtbcupval.h"
#include "lua.h"

#define hastocloseCfunc(n)	((n) < LUA_MULTRET)
#define codeNresults(n)		(-(n) - 3)

static StkId index2stack (lua_State *L, int idx);

LUA_API void lua_toclose (lua_State *L, int idx) {
  int nresults;
  StkId o;
  o = index2stack(L, idx);
  nresults = L->ci->nresults;
  luaF_newtbcupval(L, o);
  if (!hastocloseCfunc(nresults))
    L->ci->nresults = codeNresults(nresults);
}
