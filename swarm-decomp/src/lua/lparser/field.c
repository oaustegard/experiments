#include "LexState.h"
#include "ConsControl.h"
#include "fn_luaX_lookahead.h"
#include "recfield.h"
#include "listfield.h"
#include "expkind.h"

static void field (LexState *ls, ConsControl *cc) {
  switch (ls->t.token) {
    case 0x123: {
      if (luaX_lookahead(ls) != '=')
        listfield(ls, cc);
      else
        recfield(ls, cc);
      break;
    }
    case '[': {
      recfield(ls, cc);
      break;
    }
    default: {
      listfield(ls, cc);
      break;
    }
  }
}
