void tr_fc(struct tr_args *a)
{
	char *x0 = a->arg[0];
	char *x1 = a->arg[1];

	if (x0 && charread(&x0, c_fa) >= 0) {
		if (x1 && charread(&x1, c_fb) >= 0)
			return;
		c_fb[0] = ' ';
		c_fb[1] = 0;
	} else {
		c_fa[0] = 0;
		c_fb[0] = 0;
	}
}
