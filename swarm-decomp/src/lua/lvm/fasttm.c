#include "lstate.h"
#include "Table.h"
#include "TMS.h"
#include "fn_luaT_gettm.h"

static const TValue *fasttm (lua_State *L, Table *et, TMS e) {
  return (et == NULL) ? NULL
       : (et->flags & (1u << e)) ? NULL
       : luaT_gettm(et, e, ((TString **)((char *)L->l_G + 0x118))[e]);
}
