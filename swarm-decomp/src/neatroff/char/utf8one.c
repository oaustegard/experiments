int utf8one(const char *s) {
	int c = (unsigned char)*s;
	int n;
	if (~c & 0xc0) {
		n = c > 0;
		goto out;
	}
	n = 2;
	if (c & 0x20) {
		n = 3;
		if (c & 0x10)
			n = (c & 8) ? 1 : 4;
	}
out:
	return s[n] == 0;
}
