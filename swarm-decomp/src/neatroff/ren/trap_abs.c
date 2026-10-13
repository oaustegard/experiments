static int trap_abs(int i)
{
	int pos = tpos[i];

	if (pos < 0)
		pos = *nreg('p') + tpos[i];
	return pos;
}
