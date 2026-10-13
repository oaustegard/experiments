#include "FuncState.h"
#include "getjump.h"
#include "OpCode.h"

static void removevalues (FuncState *fs, int list) {
  for (; list != NO_JUMP; list = getjump(fs, list))
    patchtestreg(fs, list, NO_REG);
}
