#include <string.h>

static void q_beg(void)
{
	int n;

	n = out_n;
	out_mime = 1;
	memcpy(out + n, "=?UTF-8?", 8);
	out_n = n + 10;
	memcpy(out + n + 8, "B?", 2);
}
