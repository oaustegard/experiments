int f_nexttrap(void)
{
	int t = trap_pos(*nreg('d'));
	int v = t;

	if (cdiv)
		return t < 0 ? 0x3fffffff : t;
	if (t < 0 || *nreg('p') <= t)
		v = *nreg('p');
	return v - *nreg('d');
}
