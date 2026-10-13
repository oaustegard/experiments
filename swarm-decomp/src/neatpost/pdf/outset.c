void outset(char *k, char *v)
{
	if (strcmp("linewidth", k) == 0)
		pdf_linewid = strtol(v, 0, 10);
	if (strcmp("linecap", k) == 0)
		pdf_linecap = strtol(v, 0, 10);
	if (strcmp("linejoin", k) == 0)
		pdf_linejoin = strtol(v, 0, 10);
}
