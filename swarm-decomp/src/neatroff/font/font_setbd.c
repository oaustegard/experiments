void font_setbd(void *f, int b)
{
	struct font *F = f;

	F->bd = b;
}
