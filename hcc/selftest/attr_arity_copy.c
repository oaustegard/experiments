// EXPECT-ERROR: [parse]
// copy() takes exactly one argument; an empty list is rejected.
void fno_args (void);
void f (void) __attribute__ ((copy ()));
void g (void) __attribute__ ((copy (fno_args, fno_args)));
