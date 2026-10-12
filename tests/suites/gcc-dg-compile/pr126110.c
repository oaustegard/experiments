/* { dg-do compile } */
/* { dg-options "-O2 -fdump-tree-evrp" } */

struct rtx {
  int code;
};

static inline void *zero ()
{
  return 0;
}
static inline int three ()
{
  return 3;
}

int
can_combine_p (struct rtx *insn, struct rtx *elt)
{
  struct rtx *set;

  set = zero ();
  if (insn->code == three ())
    set = insn;
  else
    {
      set = elt;
      if (set == zero ())
        return 0;
    }

  return (set == zero ());

  return 0;
}

/* { dg-final { scan-tree-dump "Global Exported: set_.*1," "evrp" } } */

