#include "expdesc.h"
#include "expkind.h"

static void codestring (expdesc *e, TString *s) {
  e->f = e->t = -1;
  e->k = VKSTR;
  e->u.strval = s;
}
