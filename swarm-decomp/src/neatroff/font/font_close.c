void font_close(void *f)
{
	struct font *F = f;
	int i;

	for (i = 0; i < F->ngsub; i++)
		free(F->gsub[i].ent);
	for (i = 0; i < F->ngpos; i++)
		free(F->gpos[i].ent);
	dict_free(F->d78);
	dict_free(F->d80);
	dict_free(F->d88);
	iset_free(F->iab8);
	iset_free(F->iac0);
	iset_free(F->iac8);
	iset_free(F->iad0);
	free(F->gsub);
	free(F->gpos);
	free(F->glyph);
	free(f);
}
