#include "Table.h"
#include "TString.h"

const TValue *luaH_getstr (Table *t, TString *key) {
  if (key->tt == 4)
    return luaH_getshortstr(t, key);
  else {
    TValue ko;
    ko.value_.gc = (GCObject *)key;
    ko.tt_ = key->tt | 0x40;
    return getgeneric(t, &ko, 0);
  }
}
