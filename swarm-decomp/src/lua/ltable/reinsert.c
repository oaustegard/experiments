#include "lstate.h"
#include "Table.h"

static void reinsert (lua_State *L, Table *ot, Table *t) {
  int j;
  int size = 1 << ot->lsizenode;
  for (j = 0; j < size; j++) {
    Node *old = &ot->node[j];
    if (old->i_val.tt_ & 0xf) {
      TValue k;
      const TValue *slot;
      k.value_ = old->u.key_val;
      k.tt_ = old->u.key_tt;
      slot = luaH_get(t, &k);
      if (slot->tt_ == 0x20)
        luaH_newkey(L, t, &k, &old->i_val);
      else {
        TValue *s = (TValue *)slot;
        s->value_ = old->i_val.value_;
        s->tt_ = old->i_val.tt_;
      }
    }
  }
}
