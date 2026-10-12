// EXPECT-EXIT: 0
// gcc defaults to PIE on Linux: __PIC__ and __PIE__ are both 2 unless -fno-pie.
#if __PIC__ != 2 || __pic__ != 2
# error "__PIC__ must default to 2"
#endif
#if __PIE__ != 2 || __pie__ != 2
# error "__PIE__ must default to 2"
#endif
int main(void) { return 0; }
