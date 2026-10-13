#include "lobject_tags.h"
#include "ltags.h"
#include "fn_luaG_forerror.h"
#include "fn_luaV_tonumber_.h"
#include "fn_luaV_tointeger.h"

static int forlimit (lua_State *L, lua_Integer init, const TValue *lim, lua_Integer *p, lua_Integer step) {
  if (!luaV_tointeger(lim, p, (step < 0 ? 2 : 1))) {
    lua_Number flim;
    if (!(ttisfloat(lim) ? (flim = fltvalue(lim), 1) : luaV_tonumber_(lim, &flim)))
      luaG_forerror(L, lim, "limit");
    if (0 < flim) {
      if (step < 0) return 1;
      *p = LUA_MAXINTEGER;
    }
    else {
      if (step > 0) return 1;
      *p = LUA_MININTEGER;
    }
  }
  return (step > 0 ? init > *p : init < *p);
}
