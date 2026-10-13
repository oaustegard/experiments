static void
src_push(char *s, char **args)
{
	struct esrc *e;
	char *p;
	size_t n;
	int i;

	if (esrc_depth > 0x200)
		errdie("neateqn: macro recursion limit reached\n");
	e = calloc(sizeof(struct esrc), 1);
	e->prev = esrc;
	n = strlen(s) + 1;
	e->text = memcpy(malloc(n), s, n);
	e->nargs = args != NULL;
	if (args != NULL)
		for (i = 0; i < 10; i++) {
			p = args[i];
			e->arg[i] = p;
			if (p != NULL) {
				n = strlen(p) + 1;
				e->arg[i] = memcpy(malloc(n), p, n);
			}
		}
	esrc_depth++;
	esrc = e;
}
