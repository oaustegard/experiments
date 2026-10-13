#pragma once
#include "sbuf.h"
struct sbuf *sbuf_make(void);
void sbuf_mem(struct sbuf *sb, const void *p, int n);
void *sbuf_done(struct sbuf *sb);
