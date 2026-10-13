#include "getgclist.h"
#include "lgc_defs.h"

static GCObject **getgclist(GCObject *o) {
  switch (o->tt) {
    case 5: return &((Table *)o)->gclist;
    case 6: return &((LClosure *)o)->gclist;
    case 0x26: return &((CClosure *)o)->gclist;
    case 7: return &((Udata *)o)->gclist;
    case 8: return &((lua_State *)o)->gclist;
    case 9: return &((Proto *)o)->gclist;
    default: lua_assert(0); return NULL;
  }
}
