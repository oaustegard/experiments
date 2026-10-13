#include <ctype.h>
#include "fn_cpp_read.h"
#include "fn_mem_buf.h"
#include "fn_mem_len.h"
#include "fn_mem_put.h"

static int skipws(void)
{
	char *s;
	long n, p, q;
	int nl;
	unsigned char c;

top:
	if (off == len)
		goto refill;
	if (len <= off)
		goto chk;
ws:
	nl = 0;
	p = off;
	goto test;
loop:
	p++;
	nl = 1;
	if (p == len) {
		off = len;
		goto top;
	}
	q = p;
test:
	if (isspace(buf[p]))
		goto loop;
	if (nl)
		off = q;
	if (p == len)
		goto top;
chk:
	c = buf[off];
	if (c == '\\') {
		if (buf[off + 1] != '\n')
			return 0;
		off += 2;
		goto top;
	}
	if (c != '/')
		return 0;
	c = buf[off + 1];
	if (c == '/') {
		for (p = off + 1; p < len; p++) {
			if (buf[p] == '\n')
				break;
			if (buf[p] == '\\')
				p++;
		}
		off = p;
		goto top;
	}
	if (c != '*')
		return 0;
	off++;
	if (off >= len)
		goto top;
	p = off;
	for (;;) {
		if (buf[p] == '*' && buf[p + 1] == '/') {
			off = p + 2;
			goto top;
		}
		p++;
		if (p == len) {
			off = p;
			goto top;
		}
	}

refill:
	n = 0;
	do {
		if (cpp_read(&s, &n))
			return 1;
	} while (n == 0);
	mem_put(&tok_mem, s, n);
	buf = mem_buf(&tok_mem);
	len = mem_len(&tok_mem);
	if (len > off)
		goto ws;
	if (off == len)
		goto top;
	goto chk;
}
