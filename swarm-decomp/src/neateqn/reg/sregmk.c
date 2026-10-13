int sregmk(void)
{
	int i;
	char *name;

	if (sreg_n) {
		sreg_n--;
		i = sreg_free[sreg_n];
	} else {
		i = ++sreg_max;
	}
	name = (char *)(sreg_name + i);
	sprintf(name, "%s%02d", "", i);
	sprintf((char *)(sreg_read + i), "\\*%s", escarg(name));
	return i;
}
