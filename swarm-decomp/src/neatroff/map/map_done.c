#include "dict.h"
#include "fn_dict_free.h"
#include "map_dict.h"

void map_done(void)
{
	if (mapdict)
		dict_free(mapdict);
}
