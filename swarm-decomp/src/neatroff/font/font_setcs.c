void font_setcs(void *f, int a, int b)
{
	struct font *F = f;

	F->cs[0] = a;
	F->cs[1] = b;
}
