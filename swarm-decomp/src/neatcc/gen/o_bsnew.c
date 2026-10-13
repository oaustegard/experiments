void o_bsnew(char *name, long size, int flag)
{
	out_def(name, flag ? 20 : 4, bsslen, size);
	bsslen += (size + 15) & ~15;
}
