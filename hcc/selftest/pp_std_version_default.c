// EXPECT-EXIT: 0
#if __STDC_VERSION__ != 201710L
#error "default std must be gnu17"
#endif
int main(void) { return 0; }
