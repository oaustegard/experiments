#include "fn_luaK_concat.h"
#include "fn_luaK_code.h"
#include "exp2reg.h"
#include "patchlistaux.h"
#include "code_loadbool.h"


static void exp2reg (FuncState *fs, expdesc *e, int reg) {
  discharge2reg(fs, e, reg);
  if (e->k == VJMP)
    luaK_concat(fs, &e->t, e->u.info);
  if (e->t != e->f) {
    int final;
    int p_f = -1;
    int p_t = -1;
    if (need_value(fs, e->t) || need_value(fs, e->f)) {
      int fj = (e->k == VJMP) ? -1 : luaK_code(fs, 0x7fffff38);
      p_f = code_loadbool(fs, reg, 6);
      p_t = code_loadbool(fs, reg, 7);
      fs->lasttarget = fs->pc;
      patchlistaux(fs, fj, fs->pc, 0xff, fs->pc);
    }
    fs->lasttarget = fs->pc;
    final = fs->pc;
    patchlistaux(fs, e->f, final, reg, p_f);
    patchlistaux(fs, e->t, final, reg, p_t);
  }
  e->f = e->t = -1;
  e->u.info = reg;
  e->k = VNONRELOC;
}
