void tr_os(void)
{
	if (*nreg(map(".sv")))
		down(*nreg(map(".sv")));
	*nreg(map(".sv")) = 0;
}
