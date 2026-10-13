long i_reg(long r, long *p1, long *p2, long *p3, long *p4, long *p5)
{
	int k = r & 0xffffff;
	long ret, x;
	int t;

	*p1 = 0;
	*p2 = 0;
	*p3 = 0;
	*p4 = 0;
	*p5 = 0;
	if (r & 0x400) {
		t = r & 0x500000;
		*p1 = 0xffcf;
		*p2 = t ? 0x20 : 0xffcf;
		goto zero;
	}
	if (r & 0x10) {
		*p2 = 0xffcf;
		x = 0xffcf;
		if (r & 0x100000)
			x = k == 0x10 ? 0x20 : 8;
		*p3 = x;
		goto zero;
	}
	if (r & 0x20) {
		if (r & 0x100000) {
			*p2 = 0xffcf;
			*p3 = 8;
		} else {
			*p3 = 2;
			*p2 = 0xffcd;
		}
		goto zero;
	}
	if (r & 0x40) {
		ret = 1;
		if (r & 0x100000)
			return ret;
		if (k == 0x42) {
			*p1 = 4;
			*p2 = 1;
			x = ~*p1 & 0xffce;
		} else {
			*p1 = 1;
			*p2 = 1;
			x = k != 0x41 ? 0xffce : 0xffca;
		}
		*p3 = x;
		*p5 = 5;
		goto zero;
	}
	if (r & 0x80) {
		*p1 = 1;
		goto tail;
	}
	if (r & 0x100) {
		*p2 = k == 0x102 ? 1 : 0xffcf;
		goto zero;
	}
	if (k == 0x800) {
		*p2 = 0x80;
		*p3 = 1;
		*p4 = 2;
		*p5 = 0x82;
		goto zero;
	}
	if (k == 0x801) {
		*p2 = 0x80;
		*p3 = 0x40;
		*p4 = 2;
		*p5 = 0xc2;
		goto zero;
	}
	if (k == 0x8000) {
		*p2 = 1;
		goto zero;
	}
	if (r & 0x200) {
		t = r & 0x400000;
		*p1 = 1;
		*p2 = t ? 0 : 0xffcf;
		*p5 = 0xfc7;
		goto zero;
	}
	if (r & 0x10000) {
		t = r & 0x100000;
		*p1 = 0xffcf;
		*p2 = 0xffcf;
		*p3 = t ? 0x20 : 0xffcf;
		goto zero;
	}
	if (r & 0x20000) {
		t = r & 0x100000;
		*p2 = 0xffcf;
		*p3 = 0xffcf;
		*p4 = t ? 0x20 : 0xffcf;
		goto zero;
	}
	if (r & 0x2000) {
		*p2 = 0xffcf;
		goto zero;
	}
	ret = k != 0x1000;
	if (!(r & 0x4000))
		return ret;
tail:
	t = r & 0x100000;
	*p2 = 0xffcf;
	*p3 = t ? 8 : 0xffcf;
zero:
	return 0;
}
