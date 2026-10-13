static void readfunc(int local, int idx, int storage)
{
	struct func *f = (struct func *)funcs + idx;
	struct decl d;
	int i;

	strcpy(func_name, f->name);
	o_func_beg(func_name, f->nargs, !(storage & 1), f->flags);
	for (i = 0; i < f->nargs; i++) {
		memset(d.name, 0, sizeof d.name);
		memset(d.pre, 0, sizeof d.pre);
		d.t = f->param[i];
		*(long *)d.loc = o_arg2loc(i);
		strcpy(d.name, f->arg[i]);
		local_add(&d);
	}
	label = 0;
	label_n = 0;
	readstmt();
	o_func_end();
	func_name[0] = 0;
	locals_n = 0;
}
