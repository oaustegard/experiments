int nregmk(void)
{
	int i;
	char *name;

	if (nreg_n) {
		nreg_n--;
		i = nreg_free[nreg_n];
	} else {
		i = ++nreg_max;
	}
	name = (char *)(nreg_name + i);
	sprintf(name, "%s%02d", "", i);
	sprintf((char *)(nreg_read + i), "\\n%s", escarg(name));
	return i;
}
