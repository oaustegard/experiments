#include "cpp_macro_protos.h"

static int jumpcomment(void)
{
	long i;

	if (buf[cur] != '/')
		return 1;
	if (buf[cur + 1] == '*') {
		cur++;
		for (i = cur; i < len; i++)
			if (buf[i] == '*' && buf[i + 1] == '/') {
				cur = i + 2;
				return 0;
			}
		cur = i;
		return 1;
	}
	if (buf[cur + 1] == '/') {
		for (cur++; cur < len; cur++) {
			if (buf[cur] == '\n')
				break;
			if (buf[cur] == '\\')
				cur++;
		}
		return 0;
	}
	return 1;
}
