int tok_chops(int f)
{
	if (!tok[0] || tok_curtype == 3)
		return 1;
	if (f)
		return strchr("^~{}(),\"\n\t =:|.+-*/\\,()[]<>!", (unsigned char)tok[0]) != 0;
	return def_chopped((unsigned char)tok[0]);
}
