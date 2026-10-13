#pragma once
/* output routines of other TUs that postps calls (variadic out takes a format) */
void out(const char *fmt, ...);
void outeps(char *s, int a, int b);
void outpdf(char *s, int a, int b);
void outlink(char *s, int a, int b);
void outinfo(char *s);
void outrotate(int n);
void drawmbeg(char *s);
void drawmend(char *s);
