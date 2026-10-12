/* { dg-do compile } */

typedef __attribute__((vector_size(4*sizeof(int)))) int v4int;

void f(v4int *v, int a)
{
  a &= 1;
  *v = *v >> a;
  *v = *v == 0;
}
