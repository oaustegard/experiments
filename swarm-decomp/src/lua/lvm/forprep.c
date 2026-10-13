#include "lobject_tags.h"
#include "ltags.h"
#include "fn_luaG_forerror.h"
#include "fn_luaG_runerror.h"
#include "fn_luaV_tonumber_.h"
#include "fn_luaV_tointeger.h"

static int forprep (lua_State *L, TValue *ra) {
  TValue *pinit = ra;
  TValue *plimit = ra + 1;
  TValue *pstep = ra + 2;
  if (ttisinteger(pinit) && ttisinteger(pstep)) {
    lua_Integer init = ivalue(pinit);
    lua_Integer step = ivalue(pstep);
    lua_Integer limit;
    if (step == 0)
      luaG_runerror(L, "'for' step is zero");
    setivalue(ra + 3, init);
    if (forlimit(L, init, plimit, &limit, step))
      return 1;
    else {
      lua_Unsigned count;
      if (step > 0) {
        count = (lua_Unsigned)limit - (lua_Unsigned)init;
        if (step != 1)
          count /= (lua_Unsigned)step;
      }
      else {
        count = (lua_Unsigned)init - (lua_Unsigned)limit;
        count /= (lua_Unsigned)(-(step + 1)) + 1u;
      }
      setivalue(plimit, (lua_Integer)count);
    }
  }
  else {
    lua_Number init;
    lua_Number flimit;
    lua_Number step;
    if (!(ttisfloat(plimit) ? (flimit = fltvalue(plimit), 1) : luaV_tonumber_(plimit, &flimit)))
      luaG_forerror(L, plimit, "limit");
    if (!(ttisfloat(pstep) ? (step = fltvalue(pstep), 1) : luaV_tonumber_(pstep, &step)))
      luaG_forerror(L, pstep, "step");
    if (!(ttisfloat(pinit) ? (init = fltvalue(pinit), 1) : luaV_tonumber_(pinit, &init)))
      luaG_forerror(L, pinit, "initial value");
    if (step == 0)
      luaG_runerror(L, "'for' step is zero");
    if (0 < step ? flimit < init : init < flimit)
      return 1;
    else {
      setfltvalue(plimit, flimit);
      setfltvalue(pstep, step);
      setfltvalue(ra, init);
      setfltvalue(ra + 3, init);
    }
  }
  return 0;
}
