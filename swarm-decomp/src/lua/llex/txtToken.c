#include "txtToken.h"
#include "save.h"
#include "fn_luaX_token2str.h"
#include "fn_luaO_pushfstring.h"

static const char *txtToken(LexState *ls, int token) {
  switch (token) {
    case 0x123:
    case 0x124:
    case 0x121:
    case 0x122:
      save(ls, '\0');
      return luaO_pushfstring(ls->L, "'%s'", ls->buff->buffer);
    default:
      return luaX_token2str(ls, token);
  }
}
