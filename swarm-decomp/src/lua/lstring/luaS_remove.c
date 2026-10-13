#include "lstate.h"
#include "TString.h"

void luaS_remove (lua_State *L, TString *ts) {
  stringtable *tb = &G(L)->strt;
  int i = ts->hash & (tb->size - 1);
  TString **p = &tb->hash[i];
  while (*p != ts)
    p = &(*p)->u.hnext;
  *p = (*p)->u.hnext;
  tb->nuse--;
}
