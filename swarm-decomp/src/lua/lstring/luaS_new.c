#include <limits.h>
#include <string.h>
#include "lstate.h"
#include "TString.h"
#include "fn_luaS_new.h"
#include "fn_luaS_newlstr.h"

#define point2uint(p) ((unsigned int)((size_t)(p) & UINT_MAX))
#define getstr(ts) ((ts)->contents)

TString *luaS_new (lua_State *L, const char *str) {
  unsigned int i = point2uint(str) % STRCACHE_N;
  int j;
  TString **p = G(L)->strcache[i];
  for (j = 0; j < STRCACHE_M; j++) {
    if (strcmp(str, getstr(p[j])) == 0)
      return p[j];
  }
  for (j = STRCACHE_M - 1; j > 0; j--)
    p[j] = p[j - 1];
  p[0] = luaS_newlstr(L, str, strlen(str));
  return p[0];
}
