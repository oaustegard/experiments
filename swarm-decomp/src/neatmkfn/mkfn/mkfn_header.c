#include <stdio.h>
#include "fn_trfn_header.h"
#include "fn_trfn_cdefs.h"

void
mkfn_header(char *name)
{
	if (mkfn_dry)
		return;
	if (mkfn_trname)
		printf("name %s\n", mkfn_trname);
	if (mkfn_psname)
		printf("fontname %s\n", mkfn_psname);
	if (!mkfn_psname && name && *name)
		printf("fontname %s\n", name);
	if (mkfn_path)
		printf("fontpath %s\n", mkfn_path);
	trfn_header();
	if (mkfn_special)
		puts("special");
	trfn_cdefs();
}
