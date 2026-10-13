long dev_fontpos(void *p)
{
	long i = 0;

	while (fn_font[i] != p) {
		i++;
		if (i == 32)
			return 0;
	}
	return i;
}
