char *tok_pop(void)
{
	strcpy(tok_prev, tok);
	tok_read();
	return tok_prev[0] ? tok_prev : 0;
}
