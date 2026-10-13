static void refer(void)
{
	static char buf[1024];
	char key[256];
	int ids[256];
	char *s, *p, *q, *t, *r;
	int n, k, c;

	for (;;) {
		s = fgets(buf, 1024, stdin);
		if (s == NULL)
			return;
		if (s[0] != '.') {
			p = s;
			n = 0;
		} else {
			c = (unsigned char)s[1];
			if (c == '[') {
				/* bibliography block: .[ ... .] */
				t = s + 2;
				q = strchr(t, '\n');
				if (q == NULL)
					lnput(t, strlen(t));
				else
					lnput(t, q - t);
				n = 0;
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
				}
				if (!accumulate)
					refer_insert(ids, n);
				continue;
			}
			/* macro line: ".KEY args" where KEY is a reference macro */
			k = 0;
			c = (unsigned char)s[1];
			while (c & 0xdf) {
				key[k++] = c;
				c = (unsigned char)s[k + 1];
				if (k == 256)
					break;
			}
			key[k] = 0;
			if (c != ' ' || (!refer_refmac(refmac, key) && !refer_refmac(refmac_auth, key))) {
				p = s;
				n = 0;
			} else {
				for (q = s + 2; *q & 0xdf; q++)
					;
				c = (unsigned char)*q;
				k = q - s;
				if (c == ' ') {
					q++;
					while (*q == ' ')
						q++;
					k = q - s;
				}
				lnput(s, k);
				n = refer_cite(ids, q, refer_refmac(refmac_auth, key));
				if (*q & 0xdf && *q != '\n') {
					r = s + 1 + k;
					c = (unsigned char)*r;
					q = r;
					r++;
					while (c & 0xdf) {
						if (c == '\n')
							break;
						c = (unsigned char)*r;
						q = r;
						r++;
					}
				}
				lnput(q, -1);
				if (!accumulate)
					refer_insert(ids, n);
				continue;
			}
		}
		/* text line: copy it, expanding \*[KEY] citations */
		t = p;
		for (;;) {
			q = strchr(t, '\\');
			if (q == NULL) {
				write(1, p, strlen(p));
				break;
			}
			t = q + 1;
			if (*q != '\\')
				continue;
			if (q[1] != '*')
				continue;
			if (q[2] != '[')
				continue;
			r = q + 3;
			k = 0;
			c = (unsigned char)*r;
			while (c & 0xdf) {
				key[k++] = c;
				if (k == 256) {
					c = (unsigned char)r[1];
					break;
				}
				r++;
				c = (unsigned char)*r;
			}
			key[k] = 0;
			if (c != ' ')
				continue;
			if (refer_refmac(refmac, key) == 0 && refer_refmac(refmac_auth, key) == 0)
				continue;
			if (strchr(t, ']') == NULL)
				continue;
			r = strchr(t, ' ');
			t = r + 1;
			k = t - p;
			if (k < 0)
				k = strlen(p);
			write(1, p, k);
			n = refer_cite(ids, t, refer_refmac(refmac_auth, key));
			if (r[1] & 0xdf) {
				c = (unsigned char)r[1];
				while (c != ']') {
					c = (unsigned char)t[1];
					t++;
					if (!(c & 0xdf))
						break;
				}
			}
			p = t;
		}
		if (!accumulate && n > 0)
			refer_insert(ids, n);
	}
}
