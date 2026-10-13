#pragma once
struct sbuf {
	char *buf;
	int len;
	int size;
};
struct sbuf *sbuf_make(void);
int sbuf_printf(struct sbuf *, const char *, ...);
