#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "Upvaldesc.h"
#include "fn_luaC_barrier_.h"
#include "luaC_objbarrier.h"
#include "allocupvalue.h"
#include "singlevaraux.h"
#include "searchvar.h"
#include "searchupvalue.h"
#include "markupval.h"
#include "newupvalue.h"

static void singlevaraux (FuncState *fs, TString *n, expdesc *var, int base) {
  if (fs == NULL) {
    var->f = var->t = -1;
    var->k = VVOID;
    var->u.info = 0;
  }
  else {
    int v = searchvar(fs, n, var);
    if (v >= 0) {
      if (v == VLOCAL && !base)
        markupval(fs, var->u.var.vidx);
    }
    else {
      int idx = searchupvalue(fs, n);
      if (idx < 0) {
        singlevaraux(fs->prev, n, var, 0);
        if (var->k == VLOCAL || var->k == VUPVAL)
          idx = newupvalue(fs, n, var);
        else
          return;
      }
      var->f = var->t = -1;
      var->k = VUPVAL;
      var->u.info = idx;
    }
  }
}
