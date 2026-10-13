#include <ctype.h>
#include <string.h>

long tok_num(char *p, long *out)
{
	unsigned char c, f;
	long v, type;
	int base, x, t;
	char *e;

	f = c = *p;
	type = 0x14;
	if (f == '0' && tolower(p[1]) == 'x') {
		type = 4;
		p += 2;
		c = *p;
	}
	if (!strchr("0123456789abcdef", tolower(c))) {
		if (c == '\'') {
			if (p[1] == '\\')
				esc_char(&x, p + 1);
			else
				x = (unsigned char)p[1];
			*out = x;
			return type;
		}
		return 0;
	}
	base = type == 4 ? 16 : f == '0' ? 8 : 10;
	v = 0;
	while (c) {
		e = strchr("0123456789abcdef", tolower(c));
		if (!e)
			break;
		v = v * base + (e - "0123456789abcdef");
		c = p[1];
		p++;
	}
	*out = v;
	c = *p;
	while (c) {
		t = tolower(c);
		if (t == 'u')
			type &= ~0x10;
		else if (t == 'l')
			type = (type & 0x10) | 8;
		else
			break;
		c = p[1];
		p++;
	}
	return type;
}
