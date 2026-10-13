#include "outf.h"
#include "ps_state.h"
#include "ps_extern.h"
#include "ps_vars.h"

void out(const char *fmt, ...);

void docheader(char *title, int w, int h, int lw)
{
	ps_height = dev_res * h / 254;
	out("%%!PS-Adobe-2.0\n");
	out("%%%%Version: 1.0\n");
	if (title)
		out("%%%%Title: %s\n", title);
	out("%%%%Creator: Neatroff\n");
	out("%%%%DocumentFonts: (atend)\n");
	out("%%%%Pages: (atend)\n");
	out("%%%%EndComments\n");
	out("%%%%BeginProlog\n");
	out("/resolution %d def\n", dev_res);
	out("/pagesize [%d %d] def\n", (w * 72 + 127) / 254, (h * 72 + 127) / 254);
	out("/linewidth %d.%02d def\n\n", lw / 100, lw % 100);
	out("%s", "/setup {\n\tcounttomark 2 idiv {def} repeat pop\n\t/scaling 72 resolution div def\n\tlinewidth setlinewidth\n\t1 setlinecap\n\t0 pagesize 1 get translate\n\tscaling scaling scale\n\t0 0 moveto\n} def\n\n/pagesetup {\n\t/page exch def\n\tcurrentdict /pagedict known currentdict page known and {\n\t\tpage load pagedict exch get cvx exec\n\t} if\n} def\n\n/pdfmark where\n\t{ pop globaldict /?pdfmark /exec load put }\n\t{ globaldict begin\n\t\t/?pdfmark /pop load def\n\t\t/pdfmark /cleartomark load def\n\t\tend }\n\tifelse\n\n/t {neg} bind def\n/w {neg moveto show} bind def\n/m {neg moveto} bind def\n/g {neg moveto {glyphshow} forall} bind def\n/rgb {255 div 3 1 roll 255 div 3 1 roll 255 div 3 1 roll setrgbcolor} bind def\n/rot {/y exch def /x exch def x y neg translate rotate x neg y translate} bind def\n/done {/lastpage where {pop lastpage} if} def\n\n% caching fonts, as selectfont is supposed to be doing\n/fncache 16 dict def\n/selectfont_append { fncache exch dup findfont put } bind def\n/selectfont_cached {\n\texch dup fncache exch known not { dup selectfont_append } if\n\tfncache exch get exch scalefont setfont\n} bind def\n/f {\n\texch dup 3 1 roll scaling div selectfont_cached\n\tlinewidth mul scaling 10 mul div setlinewidth\n} bind def\n\n/savedmatrix matrix def\n/drawl {\n\tneg lineto\n} bind def\n/drawe {\n\tsavedmatrix currentmatrix pop scale\n\t.5 0 rmoveto currentpoint .5 0 rmoveto .5 0 360 arc\n\tsavedmatrix setmatrix\n} bind def\n/drawa {\n\t/dy2 exch def\n\t/dx2 exch def\n\t/dy1 exch def\n\t/dx1 exch def\n\tcurrentpoint dy1 neg add exch dx1 add exch\n\tdx1 dx1 mul dy1 dy1 mul add sqrt\n\tdy1 dx1 neg atan\n\tdy2 neg dx2 atan\n\tarc\n} bind def\n/draws {\n\t/y2 exch def\n\t/x2 exch def\n\t/y1 exch def\n\t/x1 exch def\n\t/y0 exch def\n\t/x0 exch def\n\tx0 5 x1 mul add 6 div\n\ty0 5 y1 mul add -6 div\n\tx2 5 x1 mul add 6 div\n\ty2 5 y1 mul add -6 div\n\tx1 x2 add 2 div\n\ty1 y2 add -2 div\n\tcurveto\n} bind def\n% including EPS files\n/EPSFBEG {\n\t/epsf_state save def\n\tneg translate\n\tdiv 3 1 roll div exch scale\n\tneg exch neg exch translate\n\t/dict_count countdictstack def\n\t/op_count count 1 sub def\n\tuserdict begin\n\t/showpage { } def\n\t0 setgray 0 setlinecap 1 setlinewidth 0 setlinejoin\n\t10 setmiterlimit [ ] 0 setdash newpath\n} bind def\n/EPSFEND {\n\tcount op_count sub {pop} repeat\n\tcountdictstack dict_count sub {end} repeat\n\tepsf_state restore\n} bind def\n");
	out("%%%%EndProlog\n");
	out("%%%%BeginSetup\n");
	out("<< /PageSize pagesize /ImagingBBox null >> setpagedevice\n");
	out("mark\n");
	out("setup\n");
	out("%%%%EndSetup\n");
}
