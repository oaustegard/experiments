void box_pile(struct box *b, struct box **items, int align, int vs)
{
	int arr[32][4];
	int ra = nregmk();
	int rb = nregmk();
	int n;

	for (n = 0; n < 32 && items[n]; n++)
		;
	box_beforeput(b, 0x80, 0);
	box_colinit(items, n, arr, ra, rb);
	printf(".if %s<(%sp*%du/100u) .nr %s (%sp*%du/100u)\n", nregname(rb), nreg(b->type), e_baselinesep,
	       nregname(rb), nreg(b->type), e_baselinesep);
	if (vs)
		printf(".nr %s +(%sp*%du/100u)\n", nregname(rb), nreg(b->type), vs);
	box_colput(items, n, b, align, arr + 1, ra, rb);
	if (n) {
		int i, j;

		for (i = 0; i < n; i++)
			for (j = 0; j < 4; j++)
				nregrm(arr[i][j]);
	}
	box_afterput(b, 0x80);
	box_toreg(b);
	nregrm(ra);
	nregrm(rb);
}
