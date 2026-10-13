#pragma once
#include "sbuf.h"

struct sbuf *sbuf_make(void);
void sbuf_chr(struct sbuf *sb, int c);
void sbuf_mem(struct sbuf *sb, const char *p, int n);
void sbuf_printf(struct sbuf *sb, const char *fmt, ...);
int sbuf_len(struct sbuf *sb);
char *sbuf_buf(struct sbuf *sb);
char *sbuf_done(struct sbuf *sb);
void sbuf_free(struct sbuf *sb);
void sbuf_rep(struct sbuf *sb, int c, int r);
