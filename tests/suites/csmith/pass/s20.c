/*
 * This is a RANDOMLY GENERATED PROGRAM.
 *
 * Generator: csmith 2.4.0
 * Git version: 0cdc710
 * Options:   --no-packed-struct --max-funcs 4 --max-block-depth 3 --max-expr-complexity 6 --max-struct-fields 6 --max-array-dim 2 --max-array-len-per-dim 8 --seed 20 -o /tmp/csmith-build-b1enirt7/20.c
 * Seed:      20
 */

#include "csmith.h"


static long __undefined;

/* --- Struct/Union Declarations --- */
/* --- GLOBAL VARIABLES --- */
static volatile int32_t g_3 = 0xC7CE53D8L;/* VOLATILE GLOBAL g_3 */
static volatile int32_t * const g_2 = &g_3;
static volatile int32_t *g_4 = &g_3;
static volatile int32_t g_8[4][8] = {{6L,5L,6L,6L,5L,6L,6L,5L},{5L,6L,6L,5L,6L,6L,5L,6L},{5L,5L,7L,5L,5L,7L,5L,5L},{6L,5L,6L,6L,5L,6L,6L,5L}};
static int32_t g_9[8] = {0xD7C1188EL,0xD7C1188EL,0xD7C1188EL,0xD7C1188EL,0xD7C1188EL,0xD7C1188EL,0xD7C1188EL,0xD7C1188EL};


/* --- FORWARD DECLARATIONS --- */
static const int32_t  func_1(void);


/* --- FUNCTIONS --- */
/* ------------------------------------------ */
/* 
 * reads : g_2 g_4 g_3 g_9
 * writes: g_4 g_3 g_9
 */
static const int32_t  func_1(void)
{ /* block id: 0 */
    uint64_t l_5 = 18446744073709551615UL;
    g_4 = g_2;
    (*g_4) ^= 0L;
    (*g_4) ^= l_5;
    for (l_5 = 0; (l_5 > 37); ++l_5)
    { /* block id: 6 */
        (*g_2) ^= (-1L);
        for (g_9[4] = 0; (g_9[4] >= 15); g_9[4] = safe_add_func_int32_t_s_s(g_9[4], 8))
        { /* block id: 10 */
            return l_5;
        }
    }
    return (*g_4);
}




/* ---------------------------------------- */
int main (int argc, char* argv[])
{
    int i, j;
    int print_hash_value = 0;
    if (argc == 2 && strcmp(argv[1], "1") == 0) print_hash_value = 1;
    platform_main_begin();
    crc32_gentab();
    func_1();
    transparent_crc(g_3, "g_3", print_hash_value);
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 8; j++)
        {
            transparent_crc(g_8[i][j], "g_8[i][j]", print_hash_value);
            if (print_hash_value) printf("index = [%d][%d]\n", i, j);

        }
    }
    for (i = 0; i < 8; i++)
    {
        transparent_crc(g_9[i], "g_9[i]", print_hash_value);
        if (print_hash_value) printf("index = [%d]\n", i);

    }
    platform_main_end(crc32_context ^ 0xFFFFFFFFUL, print_hash_value);
    return 0;
}

/************************ statistics *************************
XXX max struct depth: 0
breakdown:
   depth: 0, occurrence: 3
XXX total union variables: 0

XXX non-zero bitfields defined in structs: 0
XXX zero bitfields defined in structs: 0
XXX const bitfields defined in structs: 0
XXX volatile bitfields defined in structs: 0
XXX structs with bitfields in the program: 0
breakdown:
XXX full-bitfields structs in the program: 0
breakdown:
XXX times a bitfields struct's address is taken: 0
XXX times a bitfields struct on LHS: 0
XXX times a bitfields struct on RHS: 0
XXX times a single bitfield on LHS: 0
XXX times a single bitfield on RHS: 0

XXX max expression depth: 2
breakdown:
   depth: 1, occurrence: 10
   depth: 2, occurrence: 2

XXX total number of pointers: 2

XXX times a variable address is taken: 2
XXX times a pointer is dereferenced on RHS: 1
breakdown:
   depth: 1, occurrence: 1
XXX times a pointer is dereferenced on LHS: 3
breakdown:
   depth: 1, occurrence: 3
XXX times a pointer is compared with null: 0
XXX times a pointer is compared with address of another variable: 0
XXX times a pointer is compared with another pointer: 0
XXX times a pointer is qualified to be dereferenced: 5

XXX max dereference level: 1
breakdown:
   level: 0, occurrence: 0
   level: 1, occurrence: 8
XXX number of pointers point to pointers: 0
XXX number of pointers point to scalars: 2
XXX number of pointers point to structs: 0
XXX percent of pointers has null in alias set: 0
XXX average alias set size: 1

XXX times a non-volatile is read: 6
XXX times a non-volatile is write: 6
XXX times a volatile is read: 1
XXX    times read thru a pointer: 1
XXX times a volatile is write: 3
XXX    times written thru a pointer: 3
XXX times a volatile is available for access: 7
XXX percentage of non-volatile access: 75

XXX forward jumps: 0
XXX backward jumps: 0

XXX stmts: 8
XXX max block depth: 2
breakdown:
   depth: 0, occurrence: 5
   depth: 1, occurrence: 2
   depth: 2, occurrence: 1

XXX percentage a fresh-made variable is used: 60
XXX percentage an existing variable is used: 40
XXX total OOB instances added: 0
********************* end of statistics **********************/

