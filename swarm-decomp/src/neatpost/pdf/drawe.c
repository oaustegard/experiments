void drawe(int h, int v)
{
	drawquad(h, v);
	drawquad(h, -v);
	drawquad(-h, -v);
	drawquad(-h, v);
	o_h += h;
}
