#include "getgclist.h"
#include "linkgclist_.h"

static void genlink(global_State *g, GCObject *o) {
  if (getage(o) == G_TOUCHED1) {
    linkgclist_(o, getgclist(o), &g->grayagain);
    set2gray(o);
  }
  else if (getage(o) == G_TOUCHED2)
    o->marked ^= (G_TOUCHED2 ^ G_OLD);  /* changeage(o, G_TOUCHED2, G_OLD) as an xor */
}
