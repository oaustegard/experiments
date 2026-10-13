#include "expdesc.h"
#include "isCint.h"

static int isCint(expdesc *e) {
  return (e->k == VKINT && e->f == e->t &&
          (unsigned long long)e->u.ival <= 0xff);
}
