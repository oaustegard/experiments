#include "traverseudata.h"
#include "genlink.h"
#include "reallymarkobject.h"

static int traverseudata(global_State *g, Udata *u) {
  int i;
  markobjectN(g, u->metatable);
  for (i = 0; i < u->nuvalue; i++)
    markvalue(g, &u->uv[i].uv);
  genlink(g, (GCObject *)u);
  return 1 + u->nuvalue;
}
