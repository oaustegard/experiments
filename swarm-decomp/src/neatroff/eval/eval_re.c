int eval_re(char *s, int a, int b)
{
	int c = *s;

	switch (c) {
	case '+':
	case '-':
		s = s + 1;
		if (c == '+')
			return a + eval_up(&s, b);
		return a - eval_up(&s, b);
	default:
		return eval_up(&s, b);
	}
}
