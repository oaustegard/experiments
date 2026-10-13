#include "traversetable.h"
#include "traversestrongtable.h"
#include "traverseweakvalue.h"
#include "traverseephemeron.h"
#include "linkgclist_.h"
#include "fn_luaT_gettm.h"
#include <string.h>

static int traversetable(global_State *g, Table *h) {
  const char *weakkey, *weakvalue;
  const TValue *mode = (h->metatable == NULL || (h->metatable->flags & (1u << TM_MODE)))
                         ? NULL
                         : luaT_gettm(h->metatable, TM_MODE, g->tmname[TM_MODE]);
  TString *smode;
  markobjectN(g, h->metatable);
  if (mode && mode->tt_ == 0x44 &&
      (smode = (TString *)mode->value_.gc,
       weakkey = strchr(smode->contents, 'k'),
       weakvalue = strchr(smode->contents, 'v'),
       (weakkey || weakvalue))) {
    if (!weakkey)
      traverseweakvalue(g, h);
    else if (!weakvalue)
      traverseephemeron(g, h, 0);
    else
      linkgclist_((GCObject *)h, &h->gclist, &g->allweak);
  }
  else
    traversestrongtable(g, h);
  return 1 + h->alimit + 2 * (h->lastfree == NULL ? 0 : (1 << h->lsizenode));
}
