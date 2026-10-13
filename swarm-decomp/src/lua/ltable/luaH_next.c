#include "lstate.h"
#include "Table.h"

unsigned int luaH_realasize (const Table *t);

int luaH_next (lua_State *L, Table *t, TValue *key) {
  unsigned int asize = luaH_realasize(t);
  unsigned int i = findindex(L, t, key, asize);
  for (; i < asize; i++) {
    if ((t->array[i].tt_ & 0xf) != 0) {
      key->value_.i = i + 1;
      key->tt_ = 0x03;
      setobj(key + 1, &t->array[i]);
      return 1;
    }
  }
  i -= asize;
  for (; (int)i < (int)(1u << t->lsizenode); i++) {
    if ((t->node[i].i_val.tt_ & 0xf) != 0) {
      Node *n = &t->node[i];
      key->value_ = n->u.key_val;
      key->tt_ = n->u.key_tt;
      setobj(key + 1, &n->i_val);
      return 1;
    }
  }
  return 0;
}
