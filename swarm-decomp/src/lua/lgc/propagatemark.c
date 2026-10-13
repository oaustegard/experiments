#include "propagatemark.h"
#include "getgclist.h"
#include "traversetable.h"
#include "traverseudata.h"
#include "traverseLclosure.h"
#include "traverseCclosure.h"
#include "traverseproto.h"
#include "traversethread.h"

static lu_mem propagatemark(global_State *g) {
  GCObject *o = g->gray;
  o->marked |= (1 << BLACKBIT);
  g->gray = *getgclist(o);
  switch (o->tt) {
    case 5: return traversetable(g, (Table *)o);
    case 7: return traverseudata(g, (Udata *)o);
    case 6: return traverseLclosure(g, (LClosure *)o);
    case 0x26: return traverseCclosure(g, (CClosure *)o);
    case 9: return traverseproto(g, (Proto *)o);
    case 8: return traversethread(g, (lua_State *)o);
    default: return 0;
  }
}
