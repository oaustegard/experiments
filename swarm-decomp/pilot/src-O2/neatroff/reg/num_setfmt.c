#include <string.h>
#include <ctype.h>

void num_setfmt(int i, char *s)
{
	int j = 0;
	if (strchr("iIaA", *s)) {
		nregs_fmt[i] = *s;
	} else {
		while (isdigit((unsigned char)s[j]))
			j++;
		if (s[j] == 'x' || s[j] == 'X')
			nregs_fmt[i] = s[j] | ((j + 1) << 8);
		else
			nregs_fmt[i] = (j << 8) | '0';
	}
}
