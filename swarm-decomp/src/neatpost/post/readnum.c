static int readnum(int *p)
{
	int c;

	do
		c = getc(stdin);
	while (c == ' ');
	ungetc(c, stdin);
	switch (c) {
	case '+':
	case '-':
	case '0': case '1': case '2': case '3': case '4':
	case '5': case '6': case '7': case '8': case '9':
		*p = nextnum();
		return 0;
	}
	return 1;
}
