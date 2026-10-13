#include "LexState.h"
#include "FuncState.h"
#include "LHS_assign.h"
#include "expkind.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaK_reserveregs.h"

static void check_conflict (LexState *ls, struct LHS_assign *lh, expdesc *v) {
  FuncState *fs = ls->fs;
  int extra = fs->freereg;
  int conflict = 0;
  for (; lh; lh = lh->prev) {
    if (VINDEXED <= lh->v.k && lh->v.k <= VINDEXSTR) {
      if (lh->v.k == VINDEXUP) {
        if (v->k == VUPVAL && lh->v.u.ind.t == v->u.info) {
          conflict = 1;
          lh->v.k = VINDEXSTR;
          lh->v.u.ind.t = extra;
        }
      }
      else {
        if (v->k == VLOCAL && lh->v.u.ind.t == v->u.var.ridx) {
          conflict = 1;
          lh->v.u.ind.t = extra;
        }
        if (lh->v.k == VINDEXED && v->k == VLOCAL &&
            lh->v.u.ind.idx == v->u.var.ridx) {
          conflict = 1;
          lh->v.u.ind.idx = extra;
        }
      }
    }
  }
  if (conflict) {
    if (v->k == VLOCAL)
      luaK_codeABCk(fs, 0 /* OP_MOVE */, extra, v->u.var.ridx, 0, 0);
    else
      luaK_codeABCk(fs, 9 /* OP_GETUPVAL */, extra, v->u.info, 0, 0);
    luaK_reserveregs(fs, 1);
  }
}
