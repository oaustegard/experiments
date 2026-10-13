static int ren_next(void)
{
	if (ren_un > 0) {
		ren_un--;
		return ren_unbuf[ren_un];
	}
	return tr_next();
}
