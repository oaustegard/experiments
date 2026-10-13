#include <stdlib.h>
#include "fn_dir_done.h"
#include "dir_state.h"

void dir_done(void)
{
	free(rbuf);
	free(dbuf);
}
