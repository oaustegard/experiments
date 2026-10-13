#include "region.h"

void reg_done(void)
{
	free(dst_head);
	free(dst_next);
	free(loc_ptr);
	free(rgn);
	loc_ptr = 0;
	rgn = 0;
	rgn_sz = 0;
	rgn_n = 0;
	loc_n = 0;
}
