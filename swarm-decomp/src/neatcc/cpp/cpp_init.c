#include "cpp_protos.h"
#include "cpp_state.h"

int cpp_init(char *name)
{
	return include_file(name);
}
