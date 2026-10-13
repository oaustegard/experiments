#include "dumpSize.h"
#include "dumpString.h"
#include "Proto.h"
static void dumpConstants (DumpState *D, const Proto *f) {
  int i;
  int n = f->sizek;
  dumpInt(D, n);
  for (i = 0; i < n; i++) {
    const TValue *o = &f->k[i];
    int tt = o->tt_ & 0x3f;
    dumpByte(D, tt);
    switch (tt) {
      case 0x13:
        dumpNumber(D, o->value_.n);
        break;
      case 3:
        dumpInteger(D, o->value_.i);
        break;
      case 4:
      case 0x14:
        dumpString(D, (const TString *)o->value_.gc);
        break;
      default:
        break;
    }
  }
}
