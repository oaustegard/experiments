// EXPECT-ERROR: [parse]
// __thread (_Thread_local) cannot be combined with the typedef storage class.
__thread typedef int g4;
