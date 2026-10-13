#include "clearbykeys.h"
#include "iscleared.h"

static void clearbykeys(global_State *g, GCObject *l) {
  for (; l; l = ((Table *)l)->gclist) {
    Table *h = (Table *)l;
    Node *limit = h->node + (1 << h->lsizenode);
    Node *n;
    for (n = h->node; n < limit; n++) {
      GCObject *k = (n->u.key_tt & (1 << 6)) ? n->u.key_val.gc : NULL;
      if (iscleared(g, k))
        n->i_val.tt_ = 0x10;
      if ((n->i_val.tt_ & 0x0F) == 0) {
        if (n->u.key_tt & (1 << 6))
          n->u.key_tt = LUA_TDEADKEY;
      }
    }
  }
}
