static void box_colput(struct box **items, int n, struct box *b, int align, int (*arr)[4], int ra, int rb)
{
	int i;

	box_putf(b, "\\v'-%du*%su/2u'", n - 1, nreg(rb));
	for (i = 0; i < n; i++) {
		const char *s = "";

		if (align == 'c')
			box_putf(b, "\\h'%su-%su/2u'", nreg(ra), nreg(arr[i][0]));
		if (align == 'r')
			box_putf(b, "\\h'%su-%su'", nreg(ra), nreg(arr[i][0]));
		if (items[i])
			s = box_toreg(items[i]);
		box_putf(b, "\\v'%su'%s", i ? nreg(rb) : "0", s);
		if (align == 'l')
			box_putf(b, "\\h'-%su'", nreg(arr[i][0]));
		if (align == 'c')
			box_putf(b, "\\h'-%su+(%su-%su/2u)'", nreg(ra), nreg(ra), nreg(arr[i][0]));
		if (align == 'r')
			box_putf(b, "\\h'-%su'", nreg(ra));
	}
	box_putf(b, "\\v'-%du*%su/2u'\\h'%su'", n - 1, nreg(rb), nreg(ra));
}
