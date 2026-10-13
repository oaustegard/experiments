#include "llex_macros.h"
#include "save.h"
#include "lexerror.h"
#include "fn_luaO_str2num.h"
#include "lobject_tags.h"
#include "lctype.h"
#include "Zio.h"
#include "Mbuffer.h"
#include "fn_luaZ_fill.h"

static int read_numeral(LexState *ls, SemInfo *seminfo) {
  TValue obj;
  const char *expo = "Ee";
  int first = ls->current;
  save(ls, ls->current);
  next(ls);
  if (first == '0' && check_next2(ls, "xX"))
    expo = "Pp";
  for (;;) {
    if (check_next2(ls, expo))
      check_next2(ls, "-+");
    else if (lisxdigit(ls->current) || ls->current == '.') {
      save(ls, ls->current);
      next(ls);
    } else
      break;
  }
  if ((luai_ctype_[ls->current + 1] & 1)) {
    save(ls, ls->current);
    next(ls);
  }
  save(ls, '\0');
  if (luaO_str2num(ls->buff->buffer, &obj) == 0)
    lexerror(ls, "malformed number", 0x121);
  if (ttisinteger(&obj)) {
    seminfo->i = ivalue(&obj);
    return 0x122;
  } else {
    seminfo->r = obj.value_.n;
    return 0x121;
  }
}
