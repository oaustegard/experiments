#include "FuncState.h"
#include "fn_luaK_dischargevars.h"
#include "fn_luaK_exp2anyregup.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_exp2val.h"
#include "fn_luaK_indexed.h"
#include "fn_luaK_self.h"
#include "fn_luaX_next.h"
#include "fn_luaX_syntaxerror.h"
#include "check_match.h"
#include "error_expected.h"
#include "fieldsel.h"
#include "funcargs.h"
#include "singlevar.h"
#include "subexpr.h"

static void suffixedexp (LexState *ls, expdesc *v) {
  FuncState *fs = ls->fs;
  switch (ls->t.token) {
    case '(': {
      int line = ls->linenumber;
      luaX_next(ls);
      subexpr(ls, v, 0);
      check_match(ls, ')', '(', line);
      luaK_dischargevars(ls->fs, v);
      break;
    }
    case 0x123:
      singlevar(ls, v);
      break;
    default:
      luaX_syntaxerror(ls, "unexpected symbol");
  }
  for (;;) {
    switch (ls->t.token) {
      case '.':
        fieldsel(ls, v);
        break;
      case '[': {
        expdesc key;
        luaK_exp2anyregup(fs, v);
        luaX_next(ls);
        subexpr(ls, &key, 0);
        luaK_exp2val(ls->fs, &key);
        if (ls->t.token != ']')
          error_expected(ls, ']');
        luaX_next(ls);
        luaK_indexed(fs, v, &key);
        break;
      }
      case ':': {
        expdesc key;
        luaX_next(ls);
        if (ls->t.token != 0x123)
          error_expected(ls, 0x123);
        TString *ts = ls->t.seminfo.ts;
        luaX_next(ls);
        key.f = key.t = -1;
        key.k = VKSTR;
        key.u.strval = ts;
        luaK_self(fs, v, &key);
        funcargs(ls, v);
        break;
      }
      case '(': case 0x124: case '{':
        luaK_exp2nextreg(fs, v);
        funcargs(ls, v);
        break;
      default:
        return;
    }
  }
}
