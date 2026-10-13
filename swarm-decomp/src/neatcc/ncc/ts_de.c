static void ts_de(int dr)
{
	struct type *t = &ts[nts - 1];

	if (t->flags & 1) {
		if (t->ptr == 0) {
			struct array *a = (struct array *)arrays + t->arr;

			*t = a->t;
			t->ptr++;
		}
	}
	if (dr) {
		if (t->tag) {
			if (t->ptr || !(t->flags & 4))
				o_deref(t->ptr ? 8 : t->kind);
		}
	}
	t->tag = 0;
}
