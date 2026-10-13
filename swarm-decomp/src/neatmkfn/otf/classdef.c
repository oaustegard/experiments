#define U16(p) (((p)[0] << 8) | (p)[1])

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
	for (i = 0, r = t + 4; i < n; i++, r += 6) {
		s = U16(r);
		e = U16(r + 2);
		c = U16(r + 4);
		if (s > e)
			continue;
		for (j = s; j <= e; j++) {
			gl[cnt] = j;
			cls[cnt] = c;
			cnt++;
		}
	}
	return cnt;
}
