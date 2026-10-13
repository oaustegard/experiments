int main(int argc, char **argv)
{
	static char buf[1024];
	char *arg, *t;
	unsigned char *a;
	unsigned char *line, *p, *q, *d, *x, *y;
	int i, c, r, k, last;
	unsigned char ch, c1;

	for (i = 1; i < argc; i++) {
		a = argv[i];
		if (a[0] != '-' || (unsigned char)(a[1] - 'a') > 0x12) {
			fprintf(stderr, "Usage neatrefer [options] <input >output\nOptions:\n\t-p bib    \tspecify the database file\n\t-e        \taccumulate references\n\t-m        \tmerge multiple references in a single .[/.] block\n\t-i        \tinitials for authors' first and middle names\n\t-o xy     \tcitation macro (\\*[xy label])\n\t-a xy     \tauthor-year citation macro (\\*[xy label])\n\t-sa       \tsort by author last names\n");
			return 1;
		}
		switch (a[1]) {
		case 'a':
			arg = a[2] ? a + 2 : argv[++i];
			refmac_auth = arg;
			break;
		case 'e':
			accumulate = 1;
			break;
		case 'i':
			initials = 1;
			break;
		case 'm':
			multiref = 1;
			break;
		case 'o':
			arg = a[2] ? a + 2 : argv[++i];
			refmac = arg;
			break;
		case 'p':
			arg = a[2] ? a + 2 : argv[++i];
			refdb = fopen(arg, "r");
			if (refdb != NULL) {
				while (refdb != NULL) {
					line = fgets(buf, 1024, refdb);
					if (line == NULL)
						break;
					if (line[0] == '\n')
						continue;
					r = refs_n++;
					while (line != NULL && line[0] != '\n') {
						if (line[0] == '%' && line[1] >= 'A' && line[1] <= 'Z') {
							p = line + 2;
							while (isspace(*p))
								p++;
							last = -1;
							for (k = 0; p[k] != 0; k++)
								if (p[k] != '\n' && p[k] != ' ')
									last = k;
							p[last + 1] = 0;
							c = line[1];
							if (c == 'A') {
								k = refs[r].nau++;
								if (initials) {
									d = malloc(strlen(p) + 0x20);
									t = d;
									for (;;) {
										ch = *p;
										if (ch == ' ' || ch == '.') {
											p++;
											continue;
										}
										if (ch == 0)
											break;
										for (q = p; ; ) {
											c1 = *++q;
											if ((c1 & 0xdf) == 0 || c1 == '.')
												break;
										}
										if (d != t)
											*d++ = ' ';
										if (islower(ch) || c1 == 0) {
											for (x = p; x < q; x++)
												*d++ = *x;
											p = q;
											continue;
										}
										d[0] = ch;
										d[1] = '.';
										for (x = p + 1; x != q; ) {
											y = x;
											if (*x++ == '-') {
												p = y + 1;
												d += 3;
												d[-1] = '-';
												if (p < q) {
													d[0] = *p;
													d[1] = '.';
													x = p + 1;
													continue;
												}
												break;
											}
										}
										if (p >= q) {
											p = q;
											continue;
										}
										d += 2;
										p = q;
									}
									*d = 0;
								} else {
									t = sdup(p);
								}
								refs[r].au[k] = t;
							} else {
								refs[r].f[c] = sdup(p);
							}
							refs[r].seq = -1;
						}
						line = fgets(buf, 1024, refdb);
					}
				}
				fclose(refdb);
			}
			refdb = NULL;
			break;
		case 's':
			sortall = a[2] ? a[2] : argv[++i][0];
			break;
		default:
			break;
		}
	}
	refer();
	for (i = 0; i < refs_n; i++)
		for (c = 0; c < 128; c++)
			if (refs[i].f[c] != NULL)
				free(refs[i].f[c]);
	for (i = 0; i < refs_n; i++)
		for (c = 0; c < 128; c++)
			if (refs[i].au[c] != NULL)
				free(refs[i].au[c]);
	return 0;
}
