#include <stddef.h>
#include "lstate.h"
#include "TString.h"
#include "fn_luaC_newobj.h"
#include "lobject_tags.h"

TString *luaS_createlngstrobj (lua_State *L, size_t l) {
  unsigned int seed = G(L)->seed;
  TString *ts = (TString *)luaC_newobj(L, LUA_VLNGSTR, offsetof(TString, contents) + l + 1);
  ts->hash = seed;
  ts->contents[l] = '\0';
  ts->u.lnglen = l;
  ts->extra = 0;
  ts->shrlen = 0xFF;
  return ts;
}
