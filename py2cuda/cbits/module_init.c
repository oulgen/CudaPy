#include <HsFFI.h>

#define LIB_NAME "py2cuda.dylib"

static void library_init (void) __attribute__ ((constructor));
static void library_init (void)
{
    static char *argv[] = { LIB_NAME, 0}, **argv_ = argv;
    static int argc = 1;
    hs_init(&argc, &argv_);
}

static void library_exit (void) __attribute__ ((destructor));
static void library_exit (void)
{
  hs_exit();
}
