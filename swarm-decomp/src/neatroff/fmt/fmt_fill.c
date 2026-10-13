#include "fmt.h"
#include "fmt_fillwords.h"

int fmt_fill(struct fmt *f, int force)
{
	return fmt_fillwords(f, force);
}
