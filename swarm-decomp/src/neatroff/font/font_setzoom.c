void font_setzoom(void *f, int z)
{
	struct font *F = f;

	F->zoom = z;
}
