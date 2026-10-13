#include "lstate.h"
#include "ltags.h"
#include "Table.h"
#include "TString.h"
#include "fn_luaO_str2num.h"

static int l_strton (const TValue *obj, TValue *result) {
  if (!ttisstring(obj))
    return 0;
  else {
    TString *st = tsvalue(obj);
    return (luaO_str2num(st->contents, result) ==
            ((st->shrlen != 0xff) ? st->shrlen : st->u.lnglen) + 1);
  }
}
