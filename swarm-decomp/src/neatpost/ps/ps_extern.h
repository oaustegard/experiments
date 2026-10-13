#pragma once

/* symbols of other TUs used only by the ps TU (kept local so the shared types/ stay untouched) */
extern int dev_res;
char *font_name(void *f);
char *pdftext_static(char *s);
struct fglyph *dev_glyph(const char *name, int f);
int dev_fontid(struct font *f);
int font_wid(struct font *f, int n, int w);
int font_swid(struct font *f, int n);
void out(const char *fmt, ...);
