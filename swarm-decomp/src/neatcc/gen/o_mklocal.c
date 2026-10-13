void o_mklocal(long sz)
{
	long n = loc_n;
	long *off = loc_off;
	long s = loc_sz;
	long pos = loc_pos + ((sz + 7) & ~7);

	loc_pos = pos;
	if (n >= s) {
		s = s * 2;
		if (s < 0x80)
			s = 0x80;
		loc_sz = s;
		off = mextend(off, n, s, 8);
		loc_off = off;
		n = loc_n;
	}
	loc_n = n + 1;
	off[n] = pos;
}
