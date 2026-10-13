#include "cleargraylists.h"

static void cleargraylists(global_State *g) {
  g->gray = g->grayagain = NULL;
  g->weak = g->ephemeron = g->allweak = NULL;
}
