void def_set(const char *name, int val)
{
	const struct gvar *g;
	for (g = gvars; g != gvars + 30; g++)
		if (strcmp(g->name, name) == 0)
			*g->var = val;
}
