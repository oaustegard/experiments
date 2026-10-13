#include "lstate.h"
#include "fn_luaH_getshortstr.h"

const TValue *luaT_gettm (Table *events, TMS event, TString *ename) {
  const TValue *tm = luaH_getshortstr(events, ename);
  if ((tm->tt_ & 0xf) == 0) {  /* no tag method? */
    events->flags |= (lu_byte)(1u<<event);  /* cache this fact */
    return NULL;
  }
  else return tm;
}
