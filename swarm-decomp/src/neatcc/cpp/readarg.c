#include "cpp_macro_protos.h"

static void readarg(char *d)
{
	int depth = 0;
	int start = cur;
	long n;
	char c;

	while (cur < len) {
		if (depth == 0 && (buf[cur] == ',' || buf[cur] == ')'))
			break;
		if (jumpstr() == 0)
			continue;
		if (jumpcomment() == 0)
			continue;
		c = buf[cur++];
		if (c == ']' || c == ')')
			depth--;
		else if (c == '[' || c == '(' || c == '{')
			depth++;
		else if (c == '}')
			depth--;
	}
	if (d) {
		n = cur - start;
		memcpy(d, buf + start, n);
		d[n] = 0;
	}
}
