#include "fmt.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_dev_font.h"
#include "fn_font_swid.h"

void fmt_space(struct fmt *f)
{
	int a, b;

	a = *nreg(map(".ss"));
	b = *nreg('s');
	f->spacing += font_swid(dev_font(*nreg('f')), b, a);
}
