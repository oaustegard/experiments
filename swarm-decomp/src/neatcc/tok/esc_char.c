#include <ctype.h>
#include <string.h>

static int esc_char(int *out, char *p)
{
	unsigned char c;
	char *e;
	int i, v, n, x;

	if (*p != '\\') {
		*out = *p;
		return 1;
	}
	c = p[1];
	e = strchr("abefnrtv", c);
	if (e) {
		*out = "\a\b\x1b\f\n\r\t\v"[e - "abefnrtv"];
		return 2;
	}
	if (isdigit(p[1]) || c == 'x') {
		x = c == 'x';
		n = x ? 16 : 8;
		i = x + 1;
		v = 0;
		for (;;) {
			e = memchr("0123456789abcdef", tolower(p[i]), n);
			if (!e)
				break;
			v = v * n + (e - "0123456789abcdef");
			i++;
		}
		*out = v;
		return i;
	}
	*out = c;
	return 2;
}
