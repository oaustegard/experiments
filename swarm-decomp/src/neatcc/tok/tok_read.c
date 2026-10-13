#include <ctype.h>
#include <string.h>
#include "fn_mem_cut.h"
#include "fn_mem_put.h"
#include "fn_mem_putc.h"

static int tok_read(void)
{
	unsigned char c;
	const char *const *t;
	const char *s;
	char *p;
	int x;
	long i, n;

	off_pre = off;
	mem_cut(&tok, 0);
	if (skipws())
		return 1;
	p = buf + off;
	c = *p;
	if (c == '"') {
		mem_putc(&tok, '"');
		goto str;
	}
	if (isdigit(c))
		goto num;
	if (c == '\'')
		goto chr;
	if (isalnum(c) || c == '_')
		goto ident;
	if (off + 1 >= len)
		goto single;
	for (t = tok3; t != tok3 + 22; ) {
		s = *t++;
		if (c != s[0])
			continue;
		if (p[1] != s[1])
			continue;
		if (s[2] && s[2] != p[2])
			continue;
		n = strlen(s);
		off += n;
		mem_put(&tok, (char *)s, n);
		return 0;
	}
single:
	if (!strchr(";,{}()[]<>*&!=+-/%?:|^~.", c))
		return 1;
	off++;
	mem_putc(&tok, c);
	return 0;

ident:
	while (off < len) {
		c = buf[off];
		if (!isalnum(c) && c != '_')
			break;
		off++;
		mem_putc(&tok, c);
	}
	return 0;

chr:
	off++;
	n = 3;
	if (buf[off] == '\\')
		n = esc_char(&x, buf + off) + 2;
	i = 0;
	for (;;) {
		mem_putc(&tok, c);
		i++;
		if (i == n)
			break;
		c = buf[off];
		off++;
	}
	return 0;

str:
	if (buf[off] != '"') {
		mem_putc(&tok, '"');
		return 0;
	}
	off++;
	if (off >= len)
		return 1;
	for (;;) {
		c = buf[off];
		if (c == '"') {
			off++;
			if (skipws())
				return 1;
			goto str;
		}
		if (c == '\\') {
			n = esc_char(&x, buf + off);
			off += n;
			mem_putc(&tok, x);
		} else {
			off++;
			mem_putc(&tok, c);
		}
		if (off >= len)
			return 1;
	}

num:
	if (c == '0' && (buf[off + 1] & 0xdf) == 'X') {
		off++;
		mem_putc(&tok, '0');
		c = buf[off];
		off++;
		mem_putc(&tok, c);
	} else if (c == '0') {
		;
	}
	if (off >= len)
		return 0;
	for (;;) {
		c = buf[off];
		if (!strchr("0123456789abcdef", tolower(c)))
			break;
		off++;
		mem_putc(&tok, c);
		if (off >= len)
			return 0;
	}
	if (off >= len)
		return 0;
	for (;;) {
		c = buf[off];
		if (!strchr("uUlL", c))
			return 0;
		off++;
		mem_putc(&tok, c);
		if (off >= len)
			return 0;
	}
}
