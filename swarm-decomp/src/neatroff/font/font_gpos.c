static struct frule *font_gpos(struct font *f, int n, int a, int sel, int c, int d)
{
	struct frule *r;
	struct fent *e;

	e = xmalloc((long)n * 16);
	memset(e, 0, (long)n * 16);
	if (f->ngpos == f->capgpos) {
		f->capgpos = f->ngpos + 0x400;
		f->gpos = mextend(f->gpos, f->ngpos, f->capgpos, sizeof(struct frule));
	}
	r = &f->gpos[f->ngpos++];
	r->ent = e;
	r->unk8 = a;
	r->n = n;
	r->feat = sel;
	r->scrp = c;
	r->lang = d;
	return r;
}
