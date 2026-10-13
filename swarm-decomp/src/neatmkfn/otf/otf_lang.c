#define U16(p) (((p)[0] << 8) | (p)[1])

static int
otf_lang(const unsigned char *t, const unsigned char *lp, const char *script,
	const char *lang, struct otf_feat *ents, int n, int unused)
{
	const unsigned char *fl = t + U16(t + 6);
	const unsigned char *p;
	int req = U16(lp + 2);
	int nf = U16(lp + 4);
	int idx, i;

	if (req != 0xffff)
		n = otf_featrec(t, fl + 2 + 6 * req, script, lang, ents, n, lp[1]);
	if (nf) {
		p = lp + 6;
		i = 0;
		do {
			idx = U16(p);
			n = otf_featrec(t, fl + 2 + 6 * idx, script, lang, ents, n, p[1]);
			p += 2;
		} while (++i < nf);
	}
	return n;
}
