static struct frule *font_gsub(struct font *f, int n, int a, int sel, int c, int d)
{
	struct frule *r;
	struct fent *e;

	e = xmalloc((long)n * 16);
	memset(e, 0, (long)n * 16);
	if (f->ngsub == f->capgsub) {
		f->capgsub = f->ngsub + 0x400;
		f->gsub = mextend(f->gsub, f->ngsub, f->capgsub, sizeof(struct frule));
	}
	r = &f->gsub[f->ngsub++];
	r->ent = e;
	r->unk8 = a;
	r->n = n;
	r->feat = sel;
	r->scrp = c;
	r->lang = d;
	return r;
}
