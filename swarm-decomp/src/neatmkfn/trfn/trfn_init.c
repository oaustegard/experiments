void trfn_init(void)
{
	struct agl_ent *p;
	struct alt_ent *q;

	sbuf_char = sbuf_make();
	tab_agl = tab_alloc(4281);
	for (p = agl; p != agl + 4281; p++)
		tab_put(tab_agl, p->name, p->code);
	tab_alts = tab_alloc(353);
	for (q = alts; q != alts + 353; q++)
		tab_put(tab_alts, q->utf8, &q->ascii);
}
