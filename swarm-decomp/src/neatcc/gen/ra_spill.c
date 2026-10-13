static void ra_spill(int reg)
{
	long v = ra_vmap[reg];
	long l;
	long r;

	if (v >= 0) {
		val_tomem(v, reg);
		ra_vmap[reg] = -1;
	}
	l = ra_lmap[reg];
	if (l >= 0) {
		r = reg_rmap(ic_i, reg);
		if (l != r) {
			ra_lmap[reg] = -1;
			return;
		}
		loc_mem[ra_lmap[reg]]++;
		i_ins(0x8120000, 0, reg, 5, -loc_off[ra_lmap[reg]]);
		ra_lmap[reg] = -1;
	}
}
