#include "ltags.h"
#include "lstate.h"
#include "Table.h"
#include "Udata.h"
#include "TString.h"
#include "fn_luaV_tointegerns.h"
#include "fn_luaS_eqlngstr.h"
#include "fn_luaT_callTMres.h"

#define ttype(o) ((o)->tt_ & 0x0f)
#define LUA_TNUMBER 3
#define LUA_VLIGHTUSERDATA 0x02
#define LUA_VLCF 0x16
#define LUA_VFALSE 0x01
#define LUA_VTRUE 0x11
#define pvalue(o) ((o)->value_.p)
#define fvalue(o) ((o)->value_.f)
#define uvalue(o) ((Udata *)(o)->value_.gc)
/* fasttm(L, et, TM_EQ): gfasttm with tmname[TM_EQ] at l_G+0x140 */
#define EQ_FASTTM(L, et) ((et) == NULL ? NULL : ((et)->flags & (1u << TM_EQ)) ? NULL : luaT_gettm(et, TM_EQ, ((TString **)((char *)(L)->l_G + 0x118))[TM_EQ]))
#define l_isfalse(o) (((o)->tt_ == LUA_VFALSE) || (((o)->tt_ & 0xf) == 0))

int luaV_equalobj (lua_State *L, const TValue *t1, const TValue *t2) {
  const TValue *tm;
  if (ttypetag(t1) != ttypetag(t2)) {  /* not the same variant? */
    if (ttype(t1) != ttype(t2) || ttype(t1) != LUA_TNUMBER)
      return 0;  /* only numbers can be equal with different variants */
    else {  /* two numbers with different variants */
      lua_Integer i1, i2;
      return (luaV_tointegerns(t1, &i1, 0) &&
              luaV_tointegerns(t2, &i2, 0) &&
              i1 == i2);
    }
  }
  switch (ttypetag(t1)) {
    case LUA_VNIL: case LUA_VFALSE: case LUA_VTRUE:
      return 1;
    case LUA_VNUMINT:
      return (ivalue(t1) == ivalue(t2));
    case LUA_VNUMFLT:
      return (fltvalue(t1) == fltvalue(t2));
    case LUA_VLIGHTUSERDATA:
      return (pvalue(t1) == pvalue(t2));
    case LUA_VLCF:
      return (fvalue(t1) == fvalue(t2));
    case 4: /* short string */
      return (tsvalue(t1) == tsvalue(t2));
    case LUA_VLNGSTR:
      return luaS_eqlngstr(tsvalue(t1), tsvalue(t2));
    case 7: /* userdata */ {
      if (uvalue(t1) == uvalue(t2))
        return 1;
      else if (L == NULL)
        return 0;
      tm = EQ_FASTTM(L, uvalue(t1)->metatable);
      if (tm == NULL)
        tm = EQ_FASTTM(L, uvalue(t2)->metatable);
      break;
    }
    case 5: /* table */ {
      if (hvalue(t1) == hvalue(t2))
        return 1;
      else if (L == NULL)
        return 0;
      tm = EQ_FASTTM(L, hvalue(t1)->metatable);
      if (tm == NULL)
        tm = EQ_FASTTM(L, hvalue(t2)->metatable);
      break;
    }
    default:
      return gcvalue(t1) == gcvalue(t2);
  }
  if (tm == NULL)  /* no TM? */
    return 0;  /* objects are different */
  else {
    luaT_callTMres(L, tm, t1, t2, L->top);  /* call TM */
    return !l_isfalse(L->top);
  }
}
