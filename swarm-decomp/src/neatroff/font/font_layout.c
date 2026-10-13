int font_layout(struct font *f, struct fglyph **gl, int n, int x, struct fglyph **out, int *ix, int *a7, int *a8, int *a9, int *a10, int kn, int lg)
{
	int codes[256], res[256], tmp[256];
	int i, j, k, t, c, r, cnt, best, prev, flags, pos, acc, curs, oldkern, applied;
	int *row, *p;
	struct frule *rule;

	for (i = 0; i < n; i++)
		codes[i] = gl[i] ? gl[i] - f->glyph : -1;
	for (i = 0; i < n; i++)
		ix[i] = i;
	memset(a7, 0, n * sizeof(int));
	memset(a8, 0, n * sizeof(int));
	memset(a9, 0, n * sizeof(int));
	memset(a10, 0, n * sizeof(int));
	flags = 0;
	if (kn) {
		i = font_findfeat(f, "liga");
		if (i >= 0) {
			flags = f->feat[i];
			f->feat[i] = 1;
		}
		i = font_findfeat(f, "rlig");
		if (i >= 0) {
			flags |= f->feat[i] << 1;
			f->feat[i] = 1;
		}
	}
	cnt = n;
	prev = -1;
	while (cnt > 0) {
		best = -1;
		for (i = 0; i < cnt; i++) {
			c = codes[i];
			row = iset_get(f->iab8, c);
			if (row == NULL)
				continue;
			for (p = row; *p >= 0; p++) {
				if (prev < *p && font_rulematch(f, f->gsub + *p, codes + i, cnt - i, codes + i, i)) {
					if (best < 0 || *p < best)
						best = *p;
					break;
				}
			}
		}
		if (best < 0)
			break;
		rule = f->gsub + best;
		j = 0;
		k = 0;
		memset(tmp, 0, cnt * sizeof(int));
		do {
			tmp[k] = ix[j];
			if (font_rulematch(f, rule, codes + j, cnt - j, res + k, k)) {
				for (t = 0; t < rule->n; t++) {
					if (rule->ent[t].flags & 2)
						res[k++] = rule->ent[t].code;
					j += rule->ent[t].flags & 1;
				}
			} else {
				res[k] = codes[j];
				j++;
				k++;
			}
		} while (j < cnt);
		memcpy(codes, res, k * sizeof(int));
		memcpy(ix, tmp, k * sizeof(int));
		cnt = k;
		prev = best;
	}
	if (kn) {
		i = font_findfeat(f, "liga");
		if (i >= 0)
			f->feat[i] = flags & 1;
		i = font_findfeat(f, "rlig");
		if (i >= 0)
			f->feat[i] = (flags >> 1) & 1;
	}
	oldkern = 0;
	if (lg) {
		i = font_findfeat(f, "kern");
		if (i >= 0) {
			oldkern = f->feat[i];
			f->feat[i] = 1;
		}
	}
	curs = -1;
	if (dir_do)
		curs = font_findfeat(f, "curs");
	if (cnt > 0) {
		pos = -1;
		acc = 0;
		for (i = 0; i < cnt; i++) {
			c = codes[i];
			best = -1;
			applied = 0;
			k = -1;
			for (;;) {
				row = iset_get(f->iac0, c);
				if (row == NULL)
					break;
				for (;;) {
					r = row[k + 1];
					if (r < 0)
						break;
					k++;
					if (font_rulematch(f, f->gpos + r, codes + i, cnt - i, codes + i, i))
						break;
				}
				if (r < 0)
					break;
				rule = f->gpos + r;
				if (rule->unk8 > 0 && rule->unk8 <= best)
					continue;
				best = rule->unk8;
				for (t = 0; t < rule->n; t++) {
					a7[i + t] += rule->ent[t].aux;
					a8[i + t] += rule->ent[t].a8;
					a9[i + t] += rule->ent[t].aa;
					a10[i + t] += rule->ent[t].ac;
				}
				if (rule->feat == curs) {
					if (pos < 0)
						pos = i;
					for (t = 0; t < rule->n; t++)
						acc += rule->ent[t].ac;
					applied = 1;
				}
			}
			if (pos >= 0 && !applied) {
				a10[pos] -= acc;
				acc = 0;
				pos = -1;
			}
		}
		if (pos >= 0)
			a10[pos] -= acc;
	}
	if (lg) {
		i = font_findfeat(f, "kern");
		if (i >= 0)
			f->feat[i] = oldkern != 0;
	}
	for (i = 0; i < cnt; i++)
		out[i] = f->glyph + codes[i];
	return cnt;
}
