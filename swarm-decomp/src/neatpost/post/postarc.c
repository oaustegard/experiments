static void postarc(void)
{
	int v[4];

	if (readnum(&v[0]))
		return;
	if (readnum(&v[1]))
		return;
	if (readnum(&v[2]))
		return;
	if (readnum(&v[3]))
		return;
	drawa(v[0], v[1], v[2], v[3]);
}
