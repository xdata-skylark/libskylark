/*
 * printcnkmeminfo.hpp
 *
 *  Created on: Aug 8, 2015
 *      Author: chander
 */

#ifndef PRINTCNKMEMINFO_HPP_
#define PRINTCNKMEMINFO_HPP_

#include "spi/include/kernel/memory.h"

namespace skylark {
namespace utility {


void PrintCNKMeminfo() {

  uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;

  Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);

  printf("Allocated heap: %.2f MB, avail. heap: %.2f MB\n", (double)heap/(1024*1024), (double)heapavail/(1024*1024));
  printf("Allocated stack: %.2f MB, avail. stack: %.2f MB\n", (double)stack/(1024*1024), (double)stackavail/(1024*1024));
  printf("Memory: shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n", (double)shared/(1024*1024), (double)persist/(1024*1024), (double)guard/(1024*1024), (double)mmap/(1024*1024));

}


}

}


#endif /* PRINTCNKMEMINFO_HPP_ */
