#include "lstate.h"
#include "Table.h"

static unsigned int findindex (lua_State *L, Table *t, TValue *key, unsigned int asize) {
  unsigned int i;
  if ((key->tt_ & 0xf) == 0)
    return 0;
  i = (key->tt_ == 0x03) ? ((lua_Unsigned)key->value_.i - 1u < 0x80000000u ? (unsigned int)key->value_.i : 0) : 0;
  if (i - 1u < asize)
    return i;
  else {
    const TValue *n = getgeneric(t, key, 1);
    if (n->tt_ == 0x20)
      luaG_runerror(L, "invalid key to 'next'");
    i = (int)((Node *)n - &t->node[0]);
    return (i + 1) + asize;
  }
}
