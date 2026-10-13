#include <stddef.h>
#include "lstate.h"
#include "TString.h"
#include "fn_luaC_newobj.h"

static TString *createstrobj (lua_State *L, size_t l, int tag, unsigned int h) {
  TString *ts;
  GCObject *o;
  size_t totalsize = offsetof(TString, contents) + l + 1;
  o = luaC_newobj(L, tag, totalsize);
  ts = (TString *)o;
  ts->hash = h;
  ts->extra = 0;
  ts->contents[l] = '\0';
  return ts;
}
