// EXPECT-ERROR: [parse]
// Array designator index must have integer type; a pointer is rejected.
int a[] = { [(void *)0] = 1 };
