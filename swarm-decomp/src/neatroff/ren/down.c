static int down(int n)
{
	int d = *nreg('d');
	int k = down_amt(n);

	if (ren_traps(*nreg('d'), d + k, 1))
		return 1;
	if (-*nreg('d') > n)
		n = -*nreg('d');
	ren_sp(n, 0);
	return ren_pagelimit(0);
}
