#include "FuncState.h"
#include "freereg.h"
#include "freeregs.h"

static void freeregs(FuncState *fs, int r1, int r2) {
  if (r1 > r2) {
    freereg(fs, r1);
    freereg(fs, r2);
  }
  else {
    freereg(fs, r2);
    freereg(fs, r1);
  }
}
