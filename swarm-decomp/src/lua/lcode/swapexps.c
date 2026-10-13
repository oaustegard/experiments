#include "swapexps.h"

static void swapexps (expdesc *e1, expdesc *e2) {
  expdesc tmp = *e1;
  *e1 = *e2;
  *e2 = tmp;
}
