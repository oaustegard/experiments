#include "lstate.h"
#include "Table.h"

void luaH_resizearray (lua_State *L, Table *t, unsigned int nasize) {
  int nsize = (t->lastfree == NULL) ? 0 : (1 << t->lsizenode);
  luaH_resize(L, t, nasize, nsize);
}
