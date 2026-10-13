static void refer(void)
{
	static char buf[1024];
	char key[256];
	int ids[256];
	char *s, *p, *q, *t;
	int n, k, c;

	for (;;) {
		s = fgets(buf, 1024, stdin);
		if (s == NULL)
			return;
		if (s[0] != '.') {
			p = s;
			n = 0;
		} else if (s[1] == '[') {
			/* bibliography block: .[ ... .] */
			s += 2;
			q = strchr(s, '\n');
			if (q == NULL)
				lnput(s, strlen(s));
			else
				lnput(s, q - s);
			s = fgets(buf, 1024, stdin);
			if (s != NULL) {
				n = refer_cite(ids, s, 0);
				for (;;) {
					if (s[0] == '.' && s[1] == ']') {
						lnput(s + 2, -1);
						break;
					}
					s = fgets(buf, 1024, stdin);
					if (s == NULL)
						break;
				}
			} else {
				n = 0;
			}
			if (accumulate)
				continue;
			refer_insert(ids, n);
			continue;
		} else {
			/* macro line: ".KEY args" where KEY is a reference macro */
			k = 0;
			c = (unsigned char)s[1];
			while (c & 0xdf) {
				key[k++] = c;
				c = (unsigned char)s[k + 1];
			}
			key[k] = 0;
			if (c != ' ' || (!refer_refmac(refmac, key) && !refer_refmac(refmac_auth, key))) {
				p = s;
				n = 0;
			} else {
				p = s + 2;
				q = p;
				while (*q && *q != '\n')
					q++;
				lnput(p, q - p);
				n = refer_cite(ids, s + 2, refer_refmac(refmac_auth, key));
				if (accumulate)
					continue;
				refer_insert(ids, n);
				continue;
			}
		}
		/* text line: copy it, expanding \*[KEY] citations */
		for (;;) {
			q = strchr(p, '\\');
			if (q == NULL) {
				write(1, p, strlen(p));
				break;
			}
			t = q + 1;
			if (q[1] != '*' || q[2] != '[') {
				p = t;
				continue;
			}
			p = q;
			break;
		}
		if (accumulate)
			continue;
		if (n > 0)
			refer_insert(ids, n);
	}
}
