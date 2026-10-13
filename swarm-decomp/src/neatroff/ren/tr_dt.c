void tr_dt(struct tr_args *a)
{
	if (cdiv) {
		if (a->arg[1] == 0) {
			cdiv->f18 = -1;
			return;
		}
		cdiv->f14 = eval(a->arg[0], 'v');
		cdiv->f18 = map(a->arg[1]);
	}
}
