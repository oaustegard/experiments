static void ra_vdrop(long v)
{
	int i;
	int r;

	for (i = 0; i < 64; i++)
		if (ra_live[i] == v)
			ra_live[i] = -1;
	v = (int)v;
	for (i = 0; i < 16; i++)
		if (ra_vmap[i] == v)
			break;
	if (i == 16)
		return;
	r = -1;
	for (i = 0; i < 16; i++)
		if (ra_vmap[i] == v) {
			r = i;
			break;
		}
	ra_vmap[r] = -1;
}
