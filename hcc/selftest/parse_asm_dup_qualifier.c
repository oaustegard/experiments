// EXPECT-ERROR: [parse]
// A qualifier may appear at most once on an asm statement (gcc: duplicate 'asm' qualifier).
void f(void) {
    asm volatile volatile ("" ::: );
}
