#include "FuncState.h"
#include "expdesc.h"
#include "tonumeral.h"

static int tonumeral (const expdesc *e) {
  if (e->t != e->f)
    return 0;
  if (e->k != VKINT && e->k != VKFLT)
    return 0;
  return 1;
}
