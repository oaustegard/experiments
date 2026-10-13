#include "FuncState.h"
#include "fn_luaK_code.h"
#include "CREATE_ABCk.h"
#include "previousinstruction.h"

void luaK_nil (FuncState *fs, int from, int n) {
  int l = from + n - 1;
  Instruction *previous = previousinstruction(fs);
  if ((*previous & 0x7f) == 8) {
    int pfrom = (*previous >> 7) & 0xff;
    int pl = pfrom + ((*previous >> 16) & 0xff);
    if ((pfrom <= from) ? (from <= pl + 1) : (pfrom <= l + 1)) {
      if (pfrom < from) from = pfrom;
      if (pl > l) l = pl;
      *previous = (*previous & ~0x7f80u) | (((unsigned)from << 7) & 0x7f80u);
      *previous = (*previous & ~0xff0000u) | (((unsigned)(l - from) << 16) & 0xff0000u);
      return;
    }
  }
  luaK_code(fs, CREATE_ABCk(8, from, n - 1, 0, 0));
}
