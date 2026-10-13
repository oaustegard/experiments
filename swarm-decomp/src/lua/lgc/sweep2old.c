#include "sweep2old.h"
#include "freeobj.h"
#include "linkgclist_.h"

static void sweep2old(lua_State *L, GCObject **p) {
  GCObject *curr;
  global_State *g = G(L);
  while ((curr = *p) != NULL) {
    if (iswhite(curr)) {
      *p = curr->next;
      freeobj(L, curr);
    }
    else {
      if (curr->tt == 8) {
        lua_State *th = (lua_State *)curr;
        linkgclist_(curr, &th->gclist, &g->grayagain);
        set2gray(curr);
        setage(curr, G_OLD);
      }
      else if (curr->tt == 9 && ((UpVal *)curr)->v.p != (TValue *)&((UpVal *)curr)->u.value) {
        set2gray(curr);
        setage(curr, G_OLD);
      }
      else {
        curr->marked |= (1 << BLACKBIT);
        setage(curr, G_OLD);
      }
      p = &curr->next;
    }
  }
}
