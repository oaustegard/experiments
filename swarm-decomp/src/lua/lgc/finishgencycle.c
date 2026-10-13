#include "finishgencycle.h"
#include "correctgraylist.h"
#include "checkSizes.h"
#include "callallpendingfinalizers.h"

static void finishgencycle(lua_State *L, global_State *g) {
  GCObject **list = correctgraylist(&g->grayagain);
  *list = g->weak;
  g->weak = NULL;
  list = correctgraylist(list);
  *list = g->allweak;
  g->allweak = NULL;
  list = correctgraylist(list);
  *list = g->ephemeron;
  g->ephemeron = NULL;
  correctgraylist(list);
  checkSizes(L, g);
  g->gcstate = GCSpropagate;
  if (!g->gcemergency)
    callallpendingfinalizers(L);
}
