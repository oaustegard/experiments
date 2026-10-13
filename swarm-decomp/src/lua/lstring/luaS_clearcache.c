#include "lstate.h"

void luaS_clearcache (global_State *g) {
  int i, j;
  for (i = 0; i < 53; i++)
    for (j = 0; j < 2; j++) {
      if (g->strcache[i][j]->marked & 0x18)
        g->strcache[i][j] = g->memerrmsg;
    }
}
