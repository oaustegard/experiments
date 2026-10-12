int printf(const char *fmt, ...);
double avg(double a, double b) { return (a + b) / 2; }
int main(void) { double d = avg(1.5, 2.25); float f = 0.5f; printf("%.3f %.2f %d\n", d, f * 3, (int)(d * 100)); return 0; }
