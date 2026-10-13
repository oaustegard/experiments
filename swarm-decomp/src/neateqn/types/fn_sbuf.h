#pragma once
#include "sbuf.h"
void sbuf_init(struct sbuf *);
void sbuf_done(struct sbuf *);
char *sbuf_buf(struct sbuf *);
void sbuf_append(struct sbuf *, const char *);
