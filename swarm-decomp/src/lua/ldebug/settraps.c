#include "lua.h"
#include "lstate.h"

#define isLua(ci)	(!((ci)->callstatus & 2))

static void settraps (CallInfo *ci) {
  for (; ci != NULL; ci = ci->previous)
    if (isLua(ci))
      ci->u.l.trap = 1;
}
