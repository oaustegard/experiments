#include "FuncState.h"
#include "expdesc.h"
#include "Proto.h"
#include "isKstr.h"

static int isKstr(FuncState *fs, expdesc *e) {
  /* short literal string constant, index fits in B/C */
  return (e->k == VK && e->t == e->f && e->u.info <= 0xff &&
          fs->f->k[e->u.info].tt_ == 0x44);
}
