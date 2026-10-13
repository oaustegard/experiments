#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "getbuff.h"
#include "fn_luaG_runerror.h"
#include "fn_luaO_utf8esc.h"
#include "fn_luaS_newlstr.h"

const char *luaO_pushvfstring (lua_State *L, const char *fmt, va_list argp) {
  BuffFS buff;
  const char *e;
  buff.pushed = buff.nb = 0;
  buff.L = L;
  while ((e = strchr(fmt, '%')) != NULL) {
    addstr2buff(&buff, fmt, e - fmt);
    switch (*(e + 1)) {
      case 's': {
        const char *s = va_arg(argp, char *);
        if (s == NULL) s = "(null)";
        addstr2buff(&buff, s, strlen(s));
        break;
      }
      case 'c': {
        char c = (char)va_arg(argp, int);
        char *bf = getbuff(&buff, 1);
        *bf = c;
        buff.nb += 1;
        break;
      }
      case 'd': {
        TValue num;
        num.value_.i = va_arg(argp, int);
        num.tt_ = 3;
        addnum2buff(&buff, &num);
        break;
      }
      case 'I': {
        TValue num;
        num.value_.i = va_arg(argp, long long);
        num.tt_ = 3;
        addnum2buff(&buff, &num);
        break;
      }
      case 'f': {
        TValue num;
        num.value_.n = va_arg(argp, double);
        num.tt_ = 0x13;
        addnum2buff(&buff, &num);
        break;
      }
      case 'p': {
        const int sz = 3 * sizeof(void *) + 8;
        char *bf = getbuff(&buff, sz);
        void *p = va_arg(argp, void *);
        int len = snprintf(bf, sz, "%p", p);
        buff.nb += len;
        break;
      }
      case 'U': {
        char bf[8];
        int len = luaO_utf8esc(bf, va_arg(argp, long));
        addstr2buff(&buff, bf + 8 - len, len);
        break;
      }
      case '%': {
        char *bf = getbuff(&buff, 1);
        *bf = '%';
        buff.nb += 1;
        break;
      }
      default: {
        luaG_runerror(L, "invalid option '%%%c' to 'lua_pushfstring'", *(e + 1));
      }
    }
    fmt = e + 2;
  }
  addstr2buff(&buff, fmt, strlen(fmt));
  clearbuff(&buff);
  return (const char *)((char *)L->top[-1].value_.gc + 24);
}
