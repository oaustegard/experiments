static void val_tomem(long v, int reg)
{
	long i;

	for (i = 0; i < 64; i++)
		if (v == ra_live[i])
			break;
	if (i == 64) {
		die("neatcc: the specified value is not live\n");
		i = 0;
	}
	ra_vmax = ra_vmax > i ? ra_vmax : i + 1;
	i_ins(0x8120000, 0, reg, 5, -8 - (loc_pos + 8 * i));
}
