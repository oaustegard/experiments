// EXPECT-ERROR: specified mode too small for enumerated values
enum e { A = 256 } __attribute__((__mode__(__byte__)));
int main(void) { return A; }
