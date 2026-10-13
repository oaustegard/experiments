#include "lstate.h"
#include "Table.h"
#include "Node.h"
#include "fn_luaH_realasize.h"
#include "iscleared.h"

static void clearbyvalues (global_State *g, GCObject *l, GCObject *f) {
  for (; l != f; l = ((Table *)l)->gclist) {
    Table *h = (Table *)l;
    Node *n, *limit = h->node + (1 << h->lsizenode);
    unsigned int i;
    unsigned int asize = luaH_realasize(h);
    for (i = 0; i < asize; i++) {
      TValue *o = &h->array[i];
      GCObject *gc = (o->tt_ & 0x40) ? o->value_.gc : NULL;
      if (iscleared(g, gc))
        o->tt_ = 0x10;
    }
    for (n = h->node; n < limit; n++) {
      TValue *v = &n->i_val;
      GCObject *gc = (v->tt_ & 0x40) ? v->value_.gc : NULL;
      if (iscleared(g, gc))
        v->tt_ = 0x10;
      if ((v->tt_ & 0xf) == 0) {
        if (n->u.key_tt & 0x40)
          n->u.key_tt = 0xb;
      }
    }
  }
}
