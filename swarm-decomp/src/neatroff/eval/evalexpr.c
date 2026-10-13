static long evalexpr(char **sp)
{
	long r = evalatom(sp);
	char *s;
	int c;

	for (;;) {
		s = *sp;
		c = *s;
		if (c == 0)
			return r;
		switch (c) {
		case '*':
			*sp = s + 1;
			r *= evalatom(sp);
			break;
		case '+':
			*sp = s + 1;
			r += evalatom(sp);
			break;
		case '-':
			*sp = s + 1;
			r -= evalatom(sp);
			break;
		case '/': {
			long d;
			*sp = s + 1;
			d = evalatom(sp);
			if ((int)d == 0)
				errdie("neatroff: divide by zero\n");
			r = r / (int)d;
			break;
		}
		case '%': {
			long d;
			*sp = s + 1;
			d = evalatom(sp);
			if ((int)d == 0)
				errdie("neatroff: divide by zero\n");
			r = r % (int)d;
			break;
		}
		case '>':
			*sp = s + 1;
			if (s[1] == '=') {
				*sp = s + 2;
				r = evalatom(sp) <= r;
			} else
				r = evalatom(sp) < r;
			break;
		case '<':
			*sp = s + 1;
			if (s[1] == '=') {
				*sp = s + 2;
				r = evalatom(sp) >= r;
			} else
				r = evalatom(sp) > r;
			break;
		case '=':
			*sp = s + 1;
			if (s[1] == '=')
				*sp = s + 2;
			r = evalatom(sp) == r;
			break;
		default:
			if (c == '&') {
				*sp = s + 1;
				if (r > 0)
					r = evalatom(sp) > 0;
				else
					r = 0;
				break;
			}
			if (c == ':') {
				*sp = s + 1;
				if (r > 0)
					r = 1;
				else
					r = evalatom(sp) > 0;
				break;
			}
			return r;
		}
	}
}
