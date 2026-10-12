// EXPECT-ERROR: [parse]
// A K&R parameter declaration must start with declaration specifiers (no implicit int).
int f(a)
     a;
{
    return 0;
}
