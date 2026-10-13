#include <ctype.h>
#include <string.h>

void num_setfmt(int r, char *s)
{
	int c, k, f;
	unsigned char ch;
	long i;

	c = *s;
	if (strchr("iIaA", c)) {
		nregs_fmt[r] = c;
		return;
	}
	ch = c;
	if (isdigit((unsigned char)c)) {
		i = 1;
		do {
			ch = (unsigned char)s[i];
			k = i;
			i++;
		} while (isdigit(ch));
	} else {
		k = 0;
	}
	f = (k << 8) | '0';
	if ((char)(ch & ~0x20) == 'X') {
		k++;
		f = (k << 8) | (char)ch;
	}
	nregs_fmt[r] = f;
}
