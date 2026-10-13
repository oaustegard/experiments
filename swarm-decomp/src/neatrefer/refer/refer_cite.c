static int refer_cite(int *ids, char *s, int flag)
{
	char out[256];
	char key[256];
	int n = 0;
	int i, k;
	struct ref *c;
	char *lab;
	char *kp;

	out[0] = 0;
	for (;;) {
		while (*s && strchr(" \t\n,", *s))
			s++;
		kp = key;
		while (*s && strchr(" \t\n,]", *s) == NULL)
			*kp++ = *s++;
		*kp = 0;
		if (strcmp(key, "$LIST$") == 0) {
			struct ref **tmp;
			int m = cites_n;

			tmp = malloc(m * sizeof(*tmp));
			memcpy(tmp, cites, m * sizeof(*tmp));
			if (sortall == 'a') {
				for (i = 1; i < m; i++) {
					struct ref *x = tmp[i];
					int j = i - 1;
					while (j >= 0 && strcmp(lastname(tmp[j]->au[0]), lastname(x->au[0])) > 0) {
						tmp[j + 1] = tmp[j];
						j--;
					}
					tmp[j + 1] = x;
				}
			}
			write(1, ".]<\n", 4);
			for (i = 0; i < cites_n; i++) {
				ref_ins(tmp[i], tmp[i]->seq + 1);
			}
			write(1, ".]>", 3);
			free(tmp);
			if (*s == 0)
				return n;
			continue;
		}
		i = -1;
		for (k = 0; k < refs_n; k++) {
			if (refs[k].f['L'] != NULL && strcmp(key, refs[k].f['L']) == 0) {
				i = k;
				break;
			}
		}
		if (i < 0) {
			ids[n] = -1;
			fprintf(stderr, "refer: <%s> not found\n", key);
		} else {
			c = &refs[i];
			if (c->seq < 0) {
				c->seq = cites_n;
				cites[cites_n++] = c;
			}
			ids[n++] = c->seq;
		}
		if (*s == 0 || *s == '\n' || *s == ']')
			break;
		if (n == 0 || multiref)
			continue;
		break;
	}
	if (flag == 0) {
		qsort(ids, n, 4, intcmp);
		for (i = 0; i < n;) {
			int last = i;
			while (last + 1 < n && ids[last + 1] == ids[last] + 1)
				last++;
			if (i > 0)
				strcat(out, ",");
			if (last == i) {
				sprintf(out + strlen(out), "%d", ids[i] + 1);
			} else {
				sprintf(out + strlen(out), "%d%s%d", ids[i] + 1,
					last - i >= 2 ? "\\-" : ",", ids[last] + 1);
			}
			i = last + 1;
		}
	} else if (n > 0) {
		c = cites[ids[0]];
		sprintf(out, "%s %d", c->f['D'] ? c->f['D'] : "-", c->nau);
		for (i = 0; i < c->nau; i++) {
			char *p, *q;

			strcat(out, " ");
			lab = lastname(c->au[i]);
			p = out + strlen(out);
			if (strchr(lab, ' ') || *lab == '"') {
				q = p;
				*q++ = '"';
				for (; *lab; lab++) {
					if (*lab == '"')
						*q++ = '"';
					*q++ = *lab;
				}
				*q++ = '"';
				*q = 0;
			} else {
				strcpy(p, lab);
			}
		}
	}
	write(1, out, strlen(out));
	return n;
}
