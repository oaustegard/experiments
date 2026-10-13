#include "FuncState.h"
#include "expdesc.h"
#include "UnOpr.h"
#include "OpCode.h"
#include "fn_luaK_dischargevars.h"
#include "constfolding.h"
#include "codeunexpval.h"
#include "codenot.h"

void luaK_prefix (FuncState *fs, UnOpr opr, expdesc *e, int line) {
  static const expdesc ef = {VKINT, {0}, -1, -1};
  luaK_dischargevars(fs, e);
  switch (opr) {
    case OPR_MINUS: case OPR_BNOT:
      if (constfolding(fs, opr + 12, e, &ef))
        break;
      /* FALLTHROUGH */
    case OPR_LEN:
      codeunexpval(fs, opr + 49, e, line);
      break;
    case OPR_NOT:
      codenot(fs, e);
      break;
    default: break;
  }
}
