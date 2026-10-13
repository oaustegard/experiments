#include <string.h>
#include "fn_luaO_chunkid.h"

#define LUA_IDSIZE 60
#define RETS "..."
#define PRE "[string \""
#define POS "\"]"
#define addstr(a,b,l) ( memcpy(a,b,(l) * sizeof(char)), a += (l) )

void luaO_chunkid (char *out, const char *source, size_t srclen) {
  size_t bufflen = LUA_IDSIZE;
  if (*source == '=') {
    if (srclen <= bufflen)
      memcpy(out, source + 1, srclen * sizeof(char));
    else {
      addstr(out, source + 1, bufflen - 1);
      *out = '\0';
    }
  }
  else if (*source == '@') {
    if (srclen <= bufflen)
      memcpy(out, source + 1, srclen * sizeof(char));
    else {
      addstr(out, RETS, sizeof(RETS) - 1);
      bufflen -= sizeof(RETS) - 1;
      memcpy(out, source + 1 + srclen - bufflen, bufflen * sizeof(char));
    }
  }
  else {
    const char *nl = strchr(source, '\n');
    addstr(out, PRE, sizeof(PRE) - 1);
    bufflen -= sizeof(PRE RETS POS) - 1 + 1;
    if (srclen < bufflen && nl == NULL) {
      addstr(out, source, srclen);
    }
    else {
      if (nl != NULL) srclen = nl - source;
      if (srclen > bufflen) srclen = bufflen;
      addstr(out, source, srclen);
      addstr(out, RETS, sizeof(RETS) - 1);
    }
    memcpy(out, POS, (sizeof(POS) - 1 + 1) * sizeof(char));
  }
}
