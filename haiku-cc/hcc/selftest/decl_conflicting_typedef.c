// EXPECT-ERROR: conflicting types for 't'
typedef int t;
typedef struct { double a; int b; } t;
