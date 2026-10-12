// EXPECT-EXIT: 0
// gcc on x86-64 defaults to -fPIE: __PIC__ and __PIE__ are both 2.
#if __PIC__ != 2 || __PIE__ != 2 || __pic__ != 2 || __pie__ != 2
# error PIC/PIE macros missing
#endif
int main(void) { return 0; }
