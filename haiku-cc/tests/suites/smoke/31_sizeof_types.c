int printf(const char *fmt, ...);
int main(void) { printf("%d %d %d %d %d %d %d\n", (int)sizeof(char), (int)sizeof(short), (int)sizeof(int), (int)sizeof(long), (int)sizeof(void *), (int)sizeof(double), (int)sizeof(long long)); return 0; }
