#include "sweepgen.h"
#include "freeobj.h"

static GCObject **sweepgen(lua_State *L, global_State *g, GCObject **p, GCObject *limit, GCObject **pfirstold1) {
  static const lu_byte nextage[] = {1, 3, 3, 4, 4, 5, 6};
  int white = luaC_white(g);
  GCObject *curr;
  while ((curr = *p) != limit) {
    if (iswhite(curr)) {
      *p = curr->next;
      freeobj(L, curr);
    }
    else {
      if (getage(curr) == G_NEW)
        {
          int marked = curr->marked & ~(maskcolors | AGEBITS);
          curr->marked = cast_byte(marked | G_SURVIVAL | white);
        }
      else {
        curr->marked = cast_byte((curr->marked & ~AGEBITS) | nextage[getage(curr)]);
        if (getage(curr) == G_OLD1 && *pfirstold1 == NULL)
          *pfirstold1 = curr;
      }
      p = &curr->next;
    }
  }
  return p;
}
