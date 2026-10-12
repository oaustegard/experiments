// EXPECT-ERROR: [parse] redefinition of 'f'
static __thread int f = 1;
static __thread int f = 2;
int main(void) { return f; }
