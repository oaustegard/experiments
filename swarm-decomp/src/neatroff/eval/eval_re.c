int eval_re(char *s, int a, int b)
{
	int c = *s;

	if (c == '+' || c == '-') {
		s = s + 1;
		if (c == '+')
			return a + eval_up(&s, b);
		return a - eval_up(&s, b);
	}
	return eval_up(&s, b);
}
