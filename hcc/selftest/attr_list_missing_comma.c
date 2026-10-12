// EXPECT-ERROR: [parse]
// Attributes in one list must be separated by commas.
extern int f (void) __attribute__ ((const const));
int main(void) { return 0; }
