#include "hyph_ext.h"
#include "hy_cget.h"

static int hy_cget(char *d, char *s)
{
	char *q;
	int i, n, m;

	if (*s == '\\') {
		if (s[1] == '[') {
			s += 2;
			q = s;
			i = 0;
			if (*q == 0 || *q == ']')
				goto done;
			q = s + 1;
			i = 1;
			while (1) {
				s = q;
				d[i - 1] = q[-1];
				if (*q == 0 || *q == ']')
					break;
				q++;
				if (++i == 32) {
					i = 31;
					break;
				}
			}
	done:
			d[i] = 0;
			if (*q == 0)
				return i + 2;
			return i + 3;
		}
		else if (s[1] == 'C') {
			char dl = s[2];

			s += 3;
			q = s;
			i = 0;
			if (*q == 0 || *q == dl)
				goto done2;
			q = s + 1;
			i = 1;
			while (1) {
				s = q;
				d[i - 1] = q[-1];
				if (*q == 0 || *q == dl)
					break;
				q++;
				if (++i == 32) {
					i = 31;
					break;
				}
			}
	done2:
			d[i] = 0;
			if (*q == 0)
				return i + 3;
			return i + 4;
		}
		else if (s[1] == '(') {
			s += 2;
			n = utf8read(&s, d);
			m = utf8read(&s, d + n);
			return n + m + 2;
		}
		*d = *s++;
		return utf8read(&s, d + 1) + 1;
	}
	return utf8read(&s, d);
}
