#include "lexerror.h"
#include "fn_luaX_syntaxerror.h"

_Noreturn void luaX_syntaxerror(LexState *ls, const char *msg) {
  lexerror(ls, msg, ls->t.token);
}
