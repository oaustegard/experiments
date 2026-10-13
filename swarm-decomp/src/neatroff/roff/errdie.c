#include <stdio.h>
#include <stdlib.h>
#include "fn_errdie.h"

void errdie(char *msg)
{
	fputs(msg, stderr);
	exit(1);
}
