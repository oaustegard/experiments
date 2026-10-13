#pragma once
void sbuf_init(struct sbuf *);
void sbuf_done(struct sbuf *);
void sbuf_append(struct sbuf *, char *);
void sbuf_add(struct sbuf *, int);
char *sbuf_buf(struct sbuf *);
