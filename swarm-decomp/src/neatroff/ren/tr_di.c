void tr_di(struct tr_args *a)
{
	if (a->arg[0]) {
		char *s;

		cdiv = cdiv ? cdiv + 1 : divs;
		memset(cdiv, 0, sizeof(struct div));
		sbuf_init(&cdiv->buf);
		cdiv->f10 = map(a->arg[0]);
		cdiv->f18 = -1;
		s = (char *)a->n;
		if (s[2] == 'a') {
			if (str_get(cdiv->f10))
				sbuf_append(&cdiv->buf, str_get(cdiv->f10));
		}
		sbuf_printf(&cdiv->buf, "%c%s\n", c_cc, "\a<");
		cdiv->f20 = *nreg('d');
		cdiv->f24 = *nreg('h');
		cdiv->f28 = *nreg(map(".mk"));
		cdiv->f2c = *nreg(map(".ns"));
		*nreg('d') = 0;
		*nreg('h') = 0;
		*nreg(map(".mk")) = 0;
		*nreg(map(".ns")) = 0;
	} else if (cdiv) {
		sbuf_printf(&cdiv->buf, "%c%s\n", c_cc, "\a>");
		str_set(cdiv->f10, sbuf_buf(&cdiv->buf));
		sbuf_done(&cdiv->buf);
		*nreg(map("dl")) = cdiv->f1c;
		{
			int *p = nreg('d');
			*nreg(map("dn")) = *p;
		}
		*nreg('d') = cdiv->f20;
		*nreg('h') = cdiv->f24;
		*nreg(map(".mk")) = cdiv->f28;
		*nreg(map(".ns")) = cdiv->f2c;
		cdiv = cdiv > divs ? cdiv - 1 : 0;
	}
}
