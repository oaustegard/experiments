#include <stdio.h>
#include <string.h>

#include "c_ec.h"

void charnext_str(char *dst, char *src)
{
	int c = (unsigned char)*src;
	int n;

	if (c == c_ec || c == 4 || src[1] == 0) {
		strcpy(dst, src);
		return;
	}
	n = utf8len(c);
	if (src[n] == 0)
		strcpy(dst, src);
	else if (src[2] != 0)
		sprintf(dst, "%cC'%s'", c_ec, src);
	else if ((c & 0xc0) == 0xc0) {
		if ((c | 0xcf) != 0xff)
			sprintf(dst, "%cC'%s'", c_ec, src);
		else if (c & 8)
			sprintf(dst, "%c(%s", c_ec, src);
		else
			sprintf(dst, "%cC'%s'", c_ec, src);
	} else {
		if (c != 0)
			sprintf(dst, "%c(%s", c_ec, src);
		else
			sprintf(dst, "%cC'%s'", c_ec, src);
	}
}
