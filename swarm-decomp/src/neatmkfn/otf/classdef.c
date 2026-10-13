#define U16(p) ((int)__builtin_bswap16(*(const unsigned short *)(p)))

int
classdef(const unsigned char *t, int *gl, int *cls)
{
	int fmt, n, s, e, c;
	int i, j, cnt;
	const unsigned char *r;

	fmt = U16(t);
	cnt = 0;
	if (fmt == 1) {
		n = U16(t + 4);
		s = U16(t + 2);
		for (i = 0, r = t + 6; i < n; i++, r += 2) {
			gl[i] = s + i;
			cls[i] = U16(r);
		}
		return n;
	}
	if (fmt != 2)
		return cnt;
	n = U16(t + 2);
	for (i = 0; i < n; i++) {
		r = t + 4 + 6 * i;
		s = U16(r);
		e = U16(r + 2);
		for (j = s; j <= e; j++) {
			gl[cnt] = j;
			cls[cnt] = U16(r + 4);
			cnt++;
		}
	}
	return cnt;
}
