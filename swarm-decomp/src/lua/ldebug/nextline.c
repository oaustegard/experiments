#include "lstate.h"
#include "fn_luaG_getfuncline.h"
#include "nextline.h"

#define ABSLINEINFO	(-0x80)

static int nextline (const Proto *p, int currentline, int pc) {
  if (p->lineinfo[pc] != ABSLINEINFO)
    return currentline + p->lineinfo[pc];
  else
    return luaG_getfuncline(p, pc);
}
