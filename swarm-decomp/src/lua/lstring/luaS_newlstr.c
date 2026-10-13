#include <stddef.h>
#include <string.h>
#include "lstate.h"
#include "TString.h"
#include "fn_luaM_toobig.h"
#include "fn_luaC_newobj.h"
#include "fn_luaS_newlstr.h"
#include "internshrstr.h"

#define LUAI_MAXSHORTLEN 40
#define l_unlikely(x) __builtin_expect(((x) != 0), 0)

TString *luaS_newlstr (lua_State *L, const char *str, size_t l) {
  if (l <= LUAI_MAXSHORTLEN)
    return internshrstr(L, str, l);
  else {
    TString *ts;
    unsigned int seed;
    if (l_unlikely(l >= (0x7fffffffffffffffULL - sizeof(TString)) / sizeof(char)))
      luaM_toobig(L);
    seed = G(L)->seed;
    ts = (TString *)luaC_newobj(L, 0x14, offsetof(TString, contents) + l + 1);
    ts->hash = seed;
    ts->contents[l] = '\0';
    ts->u.lnglen = l;
    ts->shrlen = 0xff;
    ts->extra = 0;
    memcpy(ts->contents, str, l * sizeof(char));
    return ts;
  }
}
