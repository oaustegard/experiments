#include "FuncState.h"
#include "lopcodes.h"

void luaK_setoneret (FuncState *fs, expdesc *e) {
  if (e->k == VCALL) {
    e->k = VNONRELOC;
    e->u.info = (fs->f->code[e->u.info] >> POS_A) & 0xff;
  }
  else if (e->k == VVARARG) {
    fs->f->code[e->u.info] = (fs->f->code[e->u.info] & ~(0xffu << 24)) | (2u << 24);
    e->k = VRELOC;
  }
}
