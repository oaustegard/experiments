#include "traverseweakvalue.h"
#include "iscleared.h"
#include "linkgclist_.h"
#include "reallymarkobject.h"

static void traverseweakvalue(global_State *g, Table *h) {
  Node *n, *limit = h->node + (1 << h->lsizenode);
  int hasclears = (h->alimit > 0);
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
      if (!hasclears && iscleared(g, (iscollectable(&n->i_val)) ? gcvalue(&n->i_val) : NULL))
        hasclears = 1;
    }
  }
  if (g->gcstate == GCSatomic && hasclears)
    linkgclist_((GCObject *)h, &h->gclist, &g->weak);
  else
    linkgclist_((GCObject *)h, &h->gclist, &g->grayagain);
}
