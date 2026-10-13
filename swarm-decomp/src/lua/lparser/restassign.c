#include "LexState.h"
#include "FuncState.h"
#include "LHS_assign.h"
#include "expkind.h"
#include "fn_luaX_next.h"
#include "fn_luaX_syntaxerror.h"
#include "fn_luaE_incCstack.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_setoneret.h"
#include "fn_luaK_storevar.h"
#include "check_readonly.h"
#include "error_expected.h"
#include "suffixedexp.h"
#include "adjust_assign.h"
#include "OpCode.h"

static void restassign (LexState *ls, struct LHS_assign *lh, int nvars) {
  expdesc e;
  if (!(VLOCAL <= lh->v.k && lh->v.k <= VINDEXSTR))
    luaX_syntaxerror(ls, "syntax error");
  check_readonly(ls, &lh->v);
  if (ls->t.token == ',') {
    struct LHS_assign nv;
    luaX_next(ls);
    nv.prev = lh;
    suffixedexp(ls, &nv.v);
    if (!(VINDEXED <= nv.v.k && nv.v.k <= VINDEXSTR))
      check_conflict(ls, lh, &nv.v);
    luaE_incCstack(ls->L);
    restassign(ls, &nv, nvars + 1);
    ls->L->nCcalls--;
  }
  else {
    int nexps;
    if (ls->t.token != '=')
      error_expected(ls, '=');
    luaX_next(ls);
    nexps = 1;
    subexpr(ls, &e, 0);
    while (ls->t.token == ',') {
      nexps++;
      luaX_next(ls);
      luaK_exp2nextreg(ls->fs, &e);
      subexpr(ls, &e, 0);
    }
    if (nexps != nvars)
      adjust_assign(ls, nvars, nexps, &e);
    else {
      luaK_setoneret(ls->fs, &e);
      luaK_storevar(ls->fs, &lh->v, &e);
      return;
    }
  }
  e.f = e.t = NO_JUMP;
  e.k = VNONRELOC;
  e.u.info = ls->fs->freereg - 1;
  luaK_storevar(ls->fs, &lh->v, &e);
}
