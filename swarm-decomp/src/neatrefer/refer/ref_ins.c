static void ref_ins(struct ref *r, int n)
{
	char buf[4096];
	int kind = 1;
	char *p, *q, *v;
	int c, len, i;

	if (r->f['J'] == NULL) {
		if (r->f['B'] != NULL)
			kind = 3;
		else if (r->f['R'] != NULL)
			kind = 4;
		else
			kind = 2 * (r->f['I'] != NULL);
	}
	len = sprintf(buf, ".ds [F %d\n", n);
	q = buf + len;
	strcpy(q, ".]-\n");
	p = q + 4;
	*p = 0;
	if (r->nau != 0) {
		strcpy(p, ".ds [ A");
		q = p + 7;
		for (i = 0; i < r->nau; i++) {
			len = sprintf(q, "%s%s", i ? ", " : "", r->au[i]);
			q += len;
		}
		strcpy(q, "\n");
		p = q + 1;
	}
	for (c = 'B'; c != 0x5b; c++) {
		v = r->f[c];
		if (v == NULL || strchr(fields, c) == NULL)
			continue;
		len = sprintf(p, ".ds [%c %s\n", c, v);
		p += len;
		if (strchr(fields_flag, c) != NULL) {
			sprintf(p, ".nr [%c 1\n", c);
			p += 9;
		}
	}
	len = sprintf(p, ".][ %d %s\n", kind, kinds[kind]);
	p += len;
	len = p - buf;
	if (len < 0 || p < buf)
		len = strlen(buf);
	write(1, buf, len);
}
