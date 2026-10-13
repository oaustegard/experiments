#include "traverseephemeron.h"
#include "propagatemark.h"

static void convergeephemerons(global_State *g) {
  int changed;
  int dir = 0;
  do {
    GCObject *w;
    GCObject *next = g->ephemeron;
    g->ephemeron = NULL;
    changed = 0;
    while ((w = next) != NULL) {
      Table *h = (Table *)w;
      next = h->gclist;
      h->marked |= (1 << BLACKBIT);
      if (traverseephemeron(g, h, dir)) {
        while (g->gray)
          propagatemark(g);
        changed = 1;
      }
    }
    dir = !dir;
  } while (changed);
}
