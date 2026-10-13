void num_set(int i, int v)
{
	if (!nregs_fmt[i])
		nregs_fmt[i] = '0';
	*nreg(i) = v;
}
