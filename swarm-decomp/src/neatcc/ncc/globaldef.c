static void globaldef(int local, struct decl *d, int storage)
{
	struct array *ar;
	char *name = d->pre[0] ? d->pre : d->name;
	int size;

	if ((d->t.flags & 1) && !d->t.ptr) {
		ar = (struct array *)arrays + d->t.arr;
		if (!ar->size && !(storage & 2))
			ar->size = initsize();
	}
	size = type_totsz(&d->t);
	if (!(storage & 2) && (!(d->t.flags & 4) || d->t.ptr)) {
		if (tok_see()[0] == '=' && !tok_see()[1])
			*(long *)d->loc = o_dsnew(name, size, !(storage & 1));
		else
			o_bsnew(name, size, !(storage & 1));
	}
	global_add(d);
	if (tok_see()[0] == '=' && !tok_see()[1]) {
		tok_get();
		if (tok_see()[0] == '{' && !tok_see()[1]) {
			tok_get();
			initexpr(&d->t, 0, d, globalinit);
		} else {
			globalinit(d, 0, &d->t);
		}
	}
	if (tok_see()[0] == '{' && !tok_see()[1] && (d->t.flags & 4))
		readfunc(local, d->t.arr, storage);
}
