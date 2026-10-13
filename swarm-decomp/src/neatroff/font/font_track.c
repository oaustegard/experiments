void font_track(void *f, int a, int b, int c, int d)
{
	struct font *F = f;

	F->track[0] = a;
	F->track[1] = b;
	F->track[2] = c;
	F->track[3] = d;
}
