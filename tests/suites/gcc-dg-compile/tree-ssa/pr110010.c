/* { dg-additional-options -O1 } */
/* { dg-additional-options -fdump-tree-gimple } */
#define N 5

unsigned f(unsigned a, unsigned b)
{
  return (a>>N) != (b>>N);
}

unsigned f2(unsigned a, unsigned b)
{
  return (a>>N) == (b>>N);
}
/* { dg-final { scan-tree-dump-times " >> " 0 gimple } } */
