#include "iscleared.h"

static int iscleared(global_State *g, GCObject *o) {
  if (o == NULL) return 0;
  else if (novariant(o->tt) == LUA_TSTRING) {
    markobject(g, o);
    return 0;
  }
  else return iswhite(o);
}
