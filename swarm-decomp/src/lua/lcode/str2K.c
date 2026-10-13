#include "FuncState.h"
#include "expdesc.h"
#include "GCObject.h"
#include "TValue.h"
#include "addk.h"
#include "str2K.h"

static void str2K(FuncState *fs, expdesc *e) {
  TValue o;
  o.value_.gc = (GCObject *)e->u.strval;
  o.tt_ = ((GCObject *)e->u.strval)->tt | 0x40;
  e->u.info = addk(fs, &o, &o);
  e->k = VK;
}
