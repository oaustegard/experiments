#include "lstate.h"
#include "ltags.h"
#include "Table.h"
#include "TString.h"
#include "fn_luaT_gettmbyobj.h"
#include "fn_luaT_gettm.h"
#include "fn_luaT_callTMres.h"
#include "fn_luaH_getn.h"
#include "fn_luaG_typeerror.h"

/* fasttm(L, h->metatable, TM_LEN) with tmname[TM_LEN] read at l_G+0x138 */
#define TM_LEN_NAME(L) (((TString **)((char *)(L)->l_G + 0x118))[4])

void luaV_objlen (lua_State *L, StkId ra, const TValue *rb) {
  const TValue *tm;
  switch (ttypetag(rb)) {
    case (5 | (0 << 4)): {
      Table *h = hvalue(rb);
      tm = NULL;
      if (h->metatable != NULL && !(h->metatable->flags & (1u << 4)))
        tm = luaT_gettm(h->metatable, 4, TM_LEN_NAME(L));
      if (tm != NULL) break;
      setivalue(s2v(ra), luaH_getn(h));
      return;
    }
    case (4 | (1 << 4)): {
      setivalue(s2v(ra), tsvalue(rb)->u.lnglen);
      return;
    }
    case (4 | (0 << 4)): {
      setivalue(s2v(ra), tsvalue(rb)->shrlen);
      return;
    }
    default: {
      tm = luaT_gettmbyobj(L, rb, 4);
      if (__builtin_expect(ttisnil(tm) != 0, 0))
        luaG_typeerror(L, rb, "get length of");
      break;
    }
  }
  luaT_callTMres(L, tm, rb, rb, ra);
}
