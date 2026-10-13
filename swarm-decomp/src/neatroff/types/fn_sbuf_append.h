#pragma once
#include "sbuf.h"

void sbuf_add(struct sbuf *b, int c);
void sbuf_append(struct sbuf *b, char *s);
