// EXPECT-ERROR: [parse]
// A function that already has an alias/ifunc attribute may not get another one.
typedef int F (void);
static F *resolver () { return 0; }
extern int magic (void) __attribute__ ((ifunc ("resolver")));
extern int magic (void) __attribute__ ((alias ("resolver")));
