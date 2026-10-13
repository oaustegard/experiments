static void postspline(void)
{
	int v[4];
	int x, y, c;

	x = nextnum();
	y = nextnum();
	do
		c = getc(stdin);
	while (c == ' ');
	ungetc(c, stdin);
	if (c == '\n') {
		drawl(x, y);
		return;
	}
	while (readnum(&v[2]) == 0) {
		if (readnum(&v[3]) != 0)
			break;
		draws(x, y, v[2], v[3]);
		x = v[2];
		y = v[3];
	}
	draws(x, y, 0, 0);
}
