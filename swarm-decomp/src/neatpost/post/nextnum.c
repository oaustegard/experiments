#include <ctype.h>

static int nextnum(void)
{
	int c, val = 0, neg = 0;

	nextskip();
	for (;;) {
		c = getc(stdin);
		if (val == 0 && (c == '+' || c == '-')) {
			neg = c == '-';
			val = 0;
			continue;
		}
		if (!isdigit(c)) {
			ungetc(c, stdin);
			if (c < 0)
				break;
			if (!isdigit(c))
				break;
		}
		if (c < 0)
			break;
		val = val * 10 + c - '0';
	}
	return neg ? -val : val;
}
