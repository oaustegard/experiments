#pragma once
#include "elf_ehdr.h"
#include "elf_shdr.h"
/* the file header and section table written by out_write (tentative
   definitions, as symtab.h) */
static struct elf_ehdr ehdr;
static struct elf_shdr shdr[8];
