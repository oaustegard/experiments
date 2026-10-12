// EXPECT-ERROR: compound literal has function type
typedef void F (void);
void foo (void) {
    (F) {};
}
