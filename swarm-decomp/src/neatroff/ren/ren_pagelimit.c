static int ren_pagelimit(int n)
{
	int p;

	if (cdiv)
		return 0;
	n += *nreg(map("nl"));
	p = *nreg('p');
	if (n >= p) {
		ren_page(0);
		return 1;
	}
	return 0;
}
