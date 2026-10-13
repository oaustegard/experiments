#include "lstate.h"
#include "lgc_defs.h"
#include "freeobj.h"

static GCObject **sweeplistn (lua_State *L, GCObject **p, int countin, int *countout) {
  global_State *g = L->l_G;
  int ow = g->currentwhite ^ WHITEBITS;
  int white = g->currentwhite & WHITEBITS;
  int i;
  for (i = 0; *p != NULL && i < countin; i++) {
    GCObject *curr = *p;
    if (curr->marked & ow) {
      *p = curr->next;
      freeobj(L, curr);
    }
    else {
      curr->marked = cast_byte((curr->marked & ~(maskcolors | AGEBITS)) | white);
      p = &curr->next;
    }
  }
  *countout = i;
  return (*p == NULL) ? NULL : p;
}
