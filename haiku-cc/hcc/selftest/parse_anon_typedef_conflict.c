// EXPECT-ERROR: conflicting types for 't'
// Two distinct anonymous struct types cannot both be typedef'd to the same name.
typedef int t;
typedef struct { double a; } t;
