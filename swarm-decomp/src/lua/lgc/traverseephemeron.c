#include "traverseephemeron.h"
#include "iscleared.h"
#include "linkgclist_.h"
#include "genlink.h"
#include "reallymarkobject.h"

static int traverseephemeron(global_State *g, Table *h, int inv) {
  int marked = 0;
  int hasclears = 0;
  int hasww = 0;
  unsigned int i;
  unsigned int asize = luaH_realasize(h);
  int nsize = 1 << h->lsizenode;
  for (i = 0; i < asize; i++) {
    if (valiswhite(&h->array[i])) {
      marked = 1;
      reallymarkobject(g, gcvalue(&h->array[i]));
    }
  }
  for (i = 0; i < nsize; i++) {
    Node *n = inv ? &h->node[nsize - 1 - i] : &h->node[i];
    if (isempty(&n->i_val)) {
      if (n->u.key_tt & (1 << 6))
        n->u.key_tt = LUA_TDEADKEY;
    }
    else if (iscleared(g, (n->u.key_tt & (1 << 6)) ? n->u.key_val.gc : NULL)) {
      hasclears = 1;
      if (valiswhite(&n->i_val))
        hasww = 1;
    }
    else if (valiswhite(&n->i_val)) {
      marked = 1;
      reallymarkobject(g, gcvalue(&n->i_val));
    }
  }
  if (g->gcstate == GCSpropagate)
    linkgclist_((GCObject *)h, &h->gclist, &g->grayagain);
  else if (hasww)
    linkgclist_((GCObject *)h, &h->gclist, &g->ephemeron);
  else if (hasclears)
    linkgclist_((GCObject *)h, &h->gclist, &g->allweak);
  else
    genlink(g, (GCObject *)h);
  return marked;
}
