#include "reallymarkobject.h"

static void reallymarkobject(global_State *g, GCObject *o) {
  switch (o->tt) {
    case 4:
    case 0x14: {
      set2black(o);
      break;
    }
    case 0x0b: {
      UpVal *uv = (UpVal *)o;
      if (uv->v.p != &uv->u.value)
        set2gray(uv);
      else
        set2black(uv);
      markvalue(g, uv->v.p);
      break;
    }
    case 7: {
      Udata *u = (Udata *)o;
      if (u->nuvalue == 0) {
        markobjectN(g, u->metatable);
        set2black(u);
        break;
      }
    }  /* FALLTHROUGH */
    case 6:
    case 0x16:
    case 0x26:
    case 5:
    case 8:
    case 9: {
      GCObject **pnext = getgclist(o);
      linkgclist_(o, pnext, &g->gray);
      set2gray(o);
      break;
    }
    default: break;
  }
}
