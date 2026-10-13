#include "correctgraylist.h"
#include "getgclist.h"

static GCObject **correctgraylist(GCObject **p) {
  GCObject *curr;
  while ((curr = *p) != NULL) {
    GCObject **next = getgclist(curr);
    if (iswhite(curr))
      goto remove;
    if (getage(curr) == G_TOUCHED1) {
      curr->marked |= (1 << BLACKBIT);
      curr->marked ^= (G_TOUCHED1 ^ G_TOUCHED2);
      goto remain;
    }
    if (curr->tt == 8)
      goto remain;
    if (getage(curr) == G_TOUCHED2)
      curr->marked ^= (G_TOUCHED2 ^ G_OLD);
    curr->marked |= (1 << BLACKBIT);
  remove:
    *p = *next;
    continue;
  remain:
    p = next;
  }
  return p;
}
