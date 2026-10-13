static int font_rulematch(struct font *f, struct frule *r, int *in, int nin, int *out, int pos)
{
	struct fent *e = r->ent;
	int i, j, k;

	if (f->scrp >= 0 && r->scrp != f->scrp)
		return 0;
	if (r->lang >= 0 && f->lang != r->lang)
		return 0;
	if (f->feat[r->feat] == 0)
		return 0;
	for (i = 0; i < r->n && (e[i].flags & 4); i++)
		;
	if (pos < i)
		return 0;
	k = 0;
	for (j = i; j < r->n; j++) {
		if (!(e[j].flags & 2)) {
			if (k >= nin)
				return 0;
			if (e[j].flags & 8) {
				if (!iset_has(f->iad0, in[k], e[j].code))
					return 0;
			} else if (in[k] != e[j].code) {
				return 0;
			}
			k++;
		}
	}
	for (j = 0; j < r->n; j++) {
		if (!(e[j].flags & 4))
			break;
		if (e[j].flags & 8) {
			if (!iset_has(f->iad0, out[j - i], e[j].code))
				return 0;
		} else if (out[j - i] != e[j].code) {
			return 0;
		}
	}
	return 1;
}
