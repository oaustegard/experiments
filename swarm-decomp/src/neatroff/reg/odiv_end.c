#include "env.h"
#include "fn_map.h"
#include "fn_nreg.h"

void odiv_end(void)
{
	int i = --nodivs;

	*nreg(102) = odivs[i][0];
	*nreg(115) = odivs[i][1];
	*nreg(109) = odivs[i][2];
	*nreg(map(".f0")) = odivs[i][3];
	*nreg(map(".s0")) = odivs[i][4];
	*nreg(map(".m0")) = odivs[i][5];
	*nreg(map(".cd")) = odivs[i][6];
}
