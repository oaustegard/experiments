void outinfo(char *k, char *v)
{
	if (strcmp("Author", k) == 0)
		snprintf(pdf_author, 0x100, "%s", v);
	if (strcmp("Title", k) == 0)
		snprintf(pdf_title, 0x100, "%s", v);
}
