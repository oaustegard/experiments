static int tok_req(int dummy)
{
	int buf[1000];
	int k;
	int c;
	int m;
	int r = 0;

	buf[0] = c = src_next();
	if (c == '.') {
		buf[1] = c = src_next();
		if (c == ' ') {
			m = 3;
			do {
				c = src_next();
				buf[m - 1] = c;
				m++;
			} while (c == ' ');
			k = m - 1;
		} else {
			k = 2;
		}
		if (c == 'E') {
			int j = k + 1;
			c = src_next();
			buf[k] = c;
			k = j;
			r = (c == 'N');
		}
	} else {
		k = 1;
	}
	while (k > 0) {
		k--;
		src_back(buf[k]);
	}
	return r;
}
