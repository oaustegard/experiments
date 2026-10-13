#include "cpp_macro_protos.h"

static void read_tilleol(char *d)
{
	int i;

	while (cur < len && isspace(buf[cur]) && buf[cur] != '\n')
		cur++;
	while (cur < len && buf[cur] != '\n') {
		if (buf[cur] == '\\' && buf[cur + 1] == '\n') {
			cur += 2;
			continue;
		}
		i = cur;
		if (jumpstr()) {
			if (jumpcomment())
				*d++ = buf[cur++];
		} else {
			memcpy(d, buf + i, cur - i);
			d += cur - i;
		}
	}
	*d = 0;
}
