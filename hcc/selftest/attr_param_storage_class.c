// EXPECT-ERROR: [parse] storage class specified for parameter
// A parameter declaration may not carry static/extern/typedef.
void g (static int x);
