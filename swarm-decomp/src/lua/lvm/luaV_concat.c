#include <string.h>
#include "lstate.h"
#include "ltags.h"
#include "TString.h"
#include "fn_luaO_tostring.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaS_createlngstrobj.h"
#include "fn_luaT_tryconcatTM.h"
#include "fn_luaG_runerror.h"

#define l_unlikely(x) __builtin_expect(((x) != 0), 0)
#define CONCAT_SHORTLEN 40
#define CONCAT_MAXSIZE 0x7fffffffffffffdfULL

/* tostring(L, o): true if o is a string (converting numbers in place) */
#define CONCAT_TOSTR(L, o) \
  (ttisstring(o) || (ttisnumber(o) && (luaO_tostring((L), (o)), 1)))
/* isemptystr(o): short string of length 0 */
#define CONCAT_ISEMPTY(o) ((o)->tt_ == (LUA_VSHRSTR | 0x40) && tsvalue(o)->shrlen == 0)
/* vslen(o): length of a string value */
#define CONCAT_VSLEN(o) \
  (tsvalue(o)->shrlen != 0xff ? (size_t)tsvalue(o)->shrlen : tsvalue(o)->u.lnglen)

void luaV_concat (lua_State *L, int total) {
  if (total == 1)
    return;  /* "all" values already concatenated */
  do {
    StkId top = L->top;
    int n = 2;  /* number of elements handled in this pass (at least 2) */
    if (!(ttisstring(top - 2) || ttisnumber(top - 2)) ||
        !CONCAT_TOSTR(L, top - 1))
      luaT_tryconcatTM(L);
    else if (l_unlikely(CONCAT_ISEMPTY(top - 1)))  /* second operand is empty? */
      CONCAT_TOSTR(L, top - 2);  /* result is first operand */
    else if (CONCAT_ISEMPTY(top - 2)) {  /* first operand is empty string? */
      top[-2].value_ = top[-1].value_;  /* result is second op. */
      top[-2].tt_ = top[-1].tt_;
    }
    else {
      /* at least two non-empty string values; get as many as possible */
      size_t tl = CONCAT_VSLEN(top - 1);
      TString *ts;
      /* collect total length and number of strings */
      for (n = 1; n < total && CONCAT_TOSTR(L, top - n - 1); n++) {
        size_t l = CONCAT_VSLEN(top - n - 1);
        if (l_unlikely(l >= (CONCAT_MAXSIZE) - tl)) {
          L->top = top - total;  /* pop strings to avoid wasting stack */
          luaG_runerror(L, "string length overflow");
        }
        tl += l;
      }
      if (tl <= CONCAT_SHORTLEN) {  /* is result a short string? */
        char buff[CONCAT_SHORTLEN];
        size_t tl2 = 0;  /* size already copied */
        int k = n;
        do {
          TString *st = tsvalue(top - k);
          size_t l = CONCAT_VSLEN(top - k);
          memcpy(buff + tl2, st->contents, l * sizeof(char));
          tl2 += l;
        } while (--k > 0);
        ts = luaS_newlstr(L, buff, tl);
      }
      else {  /* long string; copy strings directly to uninitialized buffer */
        size_t tl2 = 0;
        int k = n;
        ts = luaS_createlngstrobj(L, tl);
        char *cbuf = ts->contents;
        do {
          TString *st = tsvalue(top - k);
          size_t l = CONCAT_VSLEN(top - k);
          memcpy(cbuf + tl2, st->contents, l * sizeof(char));
          tl2 += l;
        } while (--k > 0);
      }
      top[-n].value_.gc = (GCObject *)ts;  /* create result */
      top[-n].tt_ = ts->tt | 0x40;  /* ctb(ts->tt) */
    }
    total -= n - 1;  /* got 'n' strings to create one new */
    L->top -= n - 1;  /* popped 'n' strings and pushed one */
  } while (total > 1);  /* repeat until only 1 result left */
}
