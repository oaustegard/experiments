#include "save.h"
#include "lexerror.h"
#include "fn_luaM_saferealloc_.h"

#define MAX_SIZE ((size_t)0x7fffffffffffffff)

static void save(LexState *ls, int c) {
  Mbuffer *b = ls->buff;
  if (b->n + 1 > b->buffsize) {
    size_t newsize;
    if (b->buffsize >= MAX_SIZE / 2)
      lexerror(ls, "lexical element too long", 0);
    newsize = b->buffsize * 2;
    b->buffer = (char *)luaM_saferealloc_(ls->L, b->buffer, b->buffsize, newsize);
    b->buffsize = newsize;
  }
  b->buffer[b->n++] = (char)c;
}
