void drawc(int d)
{
	drawquad(d, d);
	drawquad(d, -d);
	drawquad(-d, -d);
	drawquad(-d, d);
	o_h += d;
}
