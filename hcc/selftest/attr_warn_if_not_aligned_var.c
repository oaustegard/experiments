// EXPECT-ERROR: [parse] 'warn_if_not_aligned' may not be specified for 'v'
int v __attribute__((warn_if_not_aligned(8)));
int main(void) { return 0; }
