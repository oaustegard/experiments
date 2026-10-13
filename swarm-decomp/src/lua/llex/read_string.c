#include "llex_macros.h"
#include "save.h"
#include "lexerror.h"
#include "esccheck.h"
#include "inclinenumber.h"
#include "readhexaesc.h"
#include "readdecesc.h"
#include "utf8esc.h"
#include "fn_luaX_newstring.h"
#include "lctype.h"

static void read_string(LexState *ls, int del, SemInfo *seminfo) {
  save_and_next(ls);
  while (ls->current != del) {
    switch (ls->current) {
    case -1:
      lexerror(ls, "unfinished string", 0x120);
      break;
    case '\n':
    case '\r':
      lexerror(ls, "unfinished string", 0x124);
      break;
    case '\\': {
      int c;
      save_and_next(ls);
      switch (ls->current) {
      case 'a': c = '\a'; goto read_save;
      case 'b': c = '\b'; goto read_save;
      case 'f': c = '\f'; goto read_save;
      case 'n': c = '\n'; goto read_save;
      case 'r': c = '\r'; goto read_save;
      case 't': c = '\t'; goto read_save;
      case 'v': c = '\v'; goto read_save;
      case 'x': c = readhexaesc(ls); goto read_save;
      case 'u': utf8esc(ls); goto no_save;
      case '\n':
      case '\r':
        inclinenumber(ls);
        c = '\n';
        goto only_save;
      case '\\':
      case '"':
      case '\'':
        c = ls->current;
        goto read_save;
      case -1: goto no_save;
      case 'z': {
        ls->buff->n -= 1;
        next(ls);
        while (lisspace(ls->current)) {
          if (currIsNewline(ls))
            inclinenumber(ls);
          else
            next(ls);
        }
        goto no_save;
      }
      default: {
        esccheck(ls, lisdigit(ls->current), "invalid escape sequence");
        c = readdecesc(ls);
        goto only_save;
      }
      }
    read_save:
      next(ls);
    only_save:
      ls->buff->n -= 1;
      save(ls, c);
    no_save:
      break;
    }
    default:
      save_and_next(ls);
    }
  }
  save_and_next(ls);
  seminfo->ts = luaX_newstring(ls, ls->buff->buffer + 1, ls->buff->n - 2);
}
