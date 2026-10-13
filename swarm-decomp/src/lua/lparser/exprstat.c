#include "LexState.h"
#include "FuncState.h"
#include "expdesc.h"
#include "LHS_assign.h"
#include "fn_luaX_syntaxerror.h"
#include "suffixedexp.h"
#include "restassign.h"

static void exprstat (LexState *ls) {
  FuncState *fs = ls->fs;
  struct LHS_assign v;
  suffixedexp(ls, &v.v);
  if (ls->t.token == '=' || ls->t.token == ',') {
    v.prev = NULL;
    restassign(ls, &v, 1);
  }
  else {
    Instruction *inst;
    if (!(v.v.k == 18))
      luaX_syntaxerror(ls, "syntax error");
    inst = &fs->f->code[v.v.u.info];
    *inst = (*inst & ~(0xffu << 16)) | (1u << 16);
  }
}
