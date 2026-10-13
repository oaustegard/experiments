#include <string.h>

static void q_end(void)
{
	int i, n;
	long k;

	if (out_mlen) {
		i = 0;
		do {
			q_put(0);
			k = i++;
		} while (out_mlen);
		n = out_n;
		memset(out + n - k - 1, '=', k + 1);
	} else {
		n = out_n;
	}
	out_mime = 0;
	out_n = n + 2;
	*(unsigned short *)(out + n) = 0x3d3f;
}
