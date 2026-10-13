#include "lstate.h"
#include "Table.h"

static void freehash (lua_State *L, Table *t) {
  if (t->lastfree != NULL)  /* not the dummy node? */
    luaM_free_(L, t->node, (size_t)(1 << t->lsizenode) * sizeof(Node));
}
