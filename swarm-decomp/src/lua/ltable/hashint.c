#include <limits.h>

static Node *hashint(const Table *t, lua_Integer i) {
  lua_Unsigned ui = (lua_Unsigned)i;
  if (ui <= (unsigned int)INT_MAX)
    return &t->node[((int)ui) % ((1 << t->lsizenode) - 1 | 1)];
  else
    return &t->node[ui % ((1 << t->lsizenode) - 1 | 1)];
}
