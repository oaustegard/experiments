long o_dsnew(char *name, long size, int flag)
{
	long *off = ds_off;
	long n = ds_n;
	long i;
	long s;
	long sz = ds_sz;

	if (n >= sz) {
		s = ds_sz * 2;
		ds_sz = s < 0x80 ? 0x80 : s;
		ds_name = mextend(ds_name, n, ds_sz, 128);
		off = mextend(ds_off, ds_n, ds_sz, 8);
		ds_off = off;
		n = ds_n;
	}
	i = (int)n;
	ds_n = n + 1;
	strcpy(ds_name + (i << 7), name);
	off[i] = mem_len(&ds);
	out_def(name, flag ? 0x12 : 2, mem_len(&ds), size);
	mem_putz(&ds, (size + 15) & ~15);
	return ds_off[i];
}
