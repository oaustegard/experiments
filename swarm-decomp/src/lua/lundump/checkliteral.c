#include "lundump.h"

static void checkliteral (LoadState *S, const char *s, const char *msg) {
  char buff[sizeof("\x1bLua") + sizeof("\x19\x93\r\n\x1a\n")];  /* larger than both */
  size_t len = strlen(s);
  loadBlock(S, buff, len);
  if (memcmp(s, buff, len) != 0)
    error(S, msg);
}
