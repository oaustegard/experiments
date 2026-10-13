void outset(char *name, char *val)
{
	if (strcmp(name, "cellht") == 0)
		c_htpt = strtol(val, 0, 10);
	if (strcmp(name, "cellwd") == 0)
		c_wdpt = strtol(val, 0, 10);
}
