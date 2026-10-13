#include "Table.h"

static int equalkey (const TValue *k1, const Node *n2, int deadok) {
  if ((k1->tt_ != n2->u.key_tt) &&
       !(deadok && n2->u.key_tt == 0xb && (k1->tt_ & (1 << 6))))
    return 0;
  switch (n2->u.key_tt) {
    case 0x00: case 0x01: case 0x11:
      return 1;
    case 0x03:
      return (k1->value_.i == n2->u.key_val.i);
    case 0x13:
      return (k1->value_.n == n2->u.key_val.n);
    case 0x02:
      return (k1->value_.p == n2->u.key_val.p);
    case 0x16:
      return (k1->value_.f == n2->u.key_val.f);
    case 0x54:
      return luaS_eqlngstr((TString *)k1->value_.gc, (TString *)n2->u.key_val.gc);
    default:
      return (k1->value_.gc == n2->u.key_val.gc);
  }
}
