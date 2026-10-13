#include "traverseLclosure.h"
#include "reallymarkobject.h"

static int traverseLclosure(global_State *g, LClosure *cl) {
  int i;
  markobjectN(g, cl->p);
  for (i = 0; i < cl->nupvalues; i++) {
    UpVal *uv = cl->upvals[i];
    markobjectN(g, uv);
  }
  return 1 + cl->nupvalues;
}
