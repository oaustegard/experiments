#include <string.h>

static void q_beg(void)
{
	int n;

	n = out_n;
	out_mime = 1;
	out[n] = '=';
	out[n + 1] = '?';
	out[n + 2] = 'U';
	out[n + 3] = 'T';
	out[n + 4] = 'F';
	out[n + 5] = '-';
	out[n + 6] = '8';
	out[n + 7] = '?';
	out_n = n + 10;
	out[n + 8] = 'B';
	out[n + 9] = '?';
}
