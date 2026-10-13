#include "LexState.h"
#include "FuncState.h"
#include "expdesc.h"
#include "ConsControl.h"
#include "subexpr.h"

static void listfield (LexState *ls, ConsControl *cc) {
  subexpr(ls, &cc->v, 0);
  cc->tostore++;
}
