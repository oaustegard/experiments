#include "llex_macros.h"
#include "llex.h"
#include "save.h"
#include "lexerror.h"
#include "skip_sep.h"
#include "inclinenumber.h"
#include "read_long_string.h"
#include "read_numeral.h"
#include "read_string.h"
#include "fn_luaX_newstring.h"
#include "lctype.h"
#include "check_next1.h"

static int llex(LexState *ls, SemInfo *seminfo) {
  ls->buff->n = 0;
  for (;;) {
    switch (ls->current) {
    case '\n':
    case '\r': {
      inclinenumber(ls);
      break;
    }
    case ' ':
    case '\f':
    case '\t':
    case '\v': {
      next(ls);
      break;
    }
    case '-': {
      next(ls);
      if (ls->current != '-')
        return '-';
      next(ls);
      if (ls->current == '[') {
        size_t sep = skip_sep(ls);
        ls->buff->n = 0;
        if (sep >= 2) {
          read_long_string(ls, NULL, sep);
          ls->buff->n = 0;
          break;
        }
      }
      while (!currIsNewline(ls) && ls->current != -1)
        next(ls);
      break;
    }
    case '[': {
      size_t sep = skip_sep(ls);
      if (sep >= 2) {
        read_long_string(ls, seminfo, sep);
        return 0x124;
      } else if (sep == 0)
        lexerror(ls, "invalid long string delimiter", 0x124);
      return '[';
    }
    case '=': {
      next(ls);
      if (check_next1(ls, '='))
        return 0x119;
      else
        return '=';
    }
    case '<': {
      next(ls);
      if (check_next1(ls, '='))
        return 0x11b;
      else if (check_next1(ls, '<'))
        return 0x11d;
      else
        return '<';
    }
    case '>': {
      next(ls);
      if (check_next1(ls, '='))
        return 0x11a;
      else if (check_next1(ls, '>'))
        return 0x11e;
      else
        return '>';
    }
    case '/': {
      next(ls);
      if (check_next1(ls, '/'))
        return 0x116;
      else
        return '/';
    }
    case '~': {
      next(ls);
      if (check_next1(ls, '='))
        return 0x11c;
      else
        return '~';
    }
    case ':': {
      next(ls);
      if (check_next1(ls, ':'))
        return 0x11f;
      else
        return ':';
    }
    case '"':
    case '\'': {
      read_string(ls, ls->current, seminfo);
      return 0x124;
    }
    case '.': {
      save_and_next(ls);
      if (check_next1(ls, '.')) {
        if (check_next1(ls, '.'))
          return 0x118;
        else
          return 0x117;
      } else if (!lisdigit(ls->current))
        return '.';
      else
        return read_numeral(ls, seminfo);
    }
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': {
      return read_numeral(ls, seminfo);
    }
    case -1: {
      return 0x120;
    }
    default: {
      if (lislalpha(ls->current)) {
        TString *ts;
        do {
          save_and_next(ls);
        } while (lislalnum(ls->current));
        ts = luaX_newstring(ls, ls->buff->buffer, ls->buff->n);
        seminfo->ts = ts;
        if (isreserved(ts))
          return ts->extra - 1 + 0x100;
        else {
          return 0x123;
        }
      } else {
        int c = ls->current;
        next(ls);
        return c;
      }
    }
    }
  }
}
