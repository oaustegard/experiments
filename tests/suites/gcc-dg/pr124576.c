/* { dg-do run } */
/* { dg-options "-std=c99" } */

int main()
{
    int i = 0;
    int j = 0;
    __typeof__((__typeof__(int[i++])*)(j++, (void*)0)) a, b, c;
    if (1 != j)
	    __builtin_abort();
}

