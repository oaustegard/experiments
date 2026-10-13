#include "getgclist.h"
#include "linkgclist_.h"

void luaC_barrierback_(lua_State *L, GCObject *o) {
  global_State *g = L->l_G;
  if (getage(o) == G_TOUCHED2)
    set2gray(o);
  else
    linkgclist_(o, getgclist(o), &g->grayagain);
  if (isold(o))
    setage(o, G_TOUCHED1);
}
