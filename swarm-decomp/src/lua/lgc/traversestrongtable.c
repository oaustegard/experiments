#include "traversestrongtable.h"
#include "genlink.h"
#include "reallymarkobject.h"
#include "fn_luaH_realasize.h"

static void traversestrongtable(global_State *g, Table *h) {
  Node *n, *limit = h->node + (1 << h->lsizenode);
  unsigned int i;
  unsigned int asize = luaH_realasize(h);
  for (i = 0; i < asize; i++)
    markvalue(g, &h->array[i]);
  for (n = h->node; n < limit; n++) {
    if (isempty(&n->i_val)) {
      if (n->u.key_tt & (1 << 6))
        n->u.key_tt = LUA_TDEADKEY;
    }
    else {
      if (n->u.key_tt & (1 << 6)) {
        GCObject *k = n->u.key_val.gc;
        if (iswhite(k))
          reallymarkobject(g, k);
      }
      markvalue(g, &n->i_val);
    }
  }
  genlink(g, (GCObject *)h);
}
