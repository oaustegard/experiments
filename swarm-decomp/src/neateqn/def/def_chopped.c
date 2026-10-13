int def_chopped(int c)
{
	if (strchr("\n {}", c))
		return 1;
	return strchr(chopped, c) != 0;
}
