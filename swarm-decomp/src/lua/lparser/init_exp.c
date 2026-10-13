#include "expdesc.h"
#include "expkind.h"

static void init_exp (expdesc *e, expkind k, int i) {
  e->f = e->t = -1;
  e->k = k;
  e->u.info = i;
}
