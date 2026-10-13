#include <string.h>
#include <time.h>
#include "lua.h"
#include "lstate.h"
#include "fn_luaS_hash.h"

#define addbuff(b,p,e) \
  { size_t t = (size_t)(e); memcpy(b + p, &t, sizeof(t)); p += sizeof(t); }

static unsigned int luai_makeseed (lua_State *L) {
  char buff[3 * sizeof(size_t)];
  unsigned int h = (unsigned int)time(NULL);
  int p = 0;
  addbuff(buff, p, L);
  addbuff(buff, p, &h);
  addbuff(buff, p, &lua_newstate);
  return luaS_hash(buff, p, h);
}
