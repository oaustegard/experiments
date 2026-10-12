// EXPECT-ERROR: [parse]
// A weakref alias declared without `static` has external linkage: rejected.
extern int target;
extern int alias __attribute__((weakref("target")));
int main(void) { return alias; }
