#include "lundump.h"

static TString *loadString (LoadState *S, Proto *p) {
  TString *ts = loadStringN(S, p);
  if (ts == NULL)
    error(S, "bad format for constant string");
  return ts;
}
