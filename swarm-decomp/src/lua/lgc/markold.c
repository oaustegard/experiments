#include "markold.h"
#include "reallymarkobject.h"

static void markold(global_State *g, GCObject *from, GCObject *to) {
  GCObject *p;
  for (p = from; p != to; p = p->next) {
    if (getage(p) == G_OLD1) {
      p->marked ^= (G_OLD1 ^ G_OLD);
      if (isblack(p))
        reallymarkobject(g, p);
    }
  }
}
