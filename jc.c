#include <stdio.h>

#include "Python.h"

#include "jc.h"

//In most cases, the easiest way to deal with this problem is to rewrite your C source 
//to use Pythonic methods, e.g. PySys_WriteStdout:
//https://github.com/ipython/ipython/issues/1230 
#define printf PySys_WriteStdout

// In C int foo() and int foo(void) are different functions.
// http://stackoverflow.com/questions/42125/function-declaration-isnt-a-prototype 
int prt( void)
{
	printf( "Hello\n");

	return 0;
}

int prt_str( char* str)
{
	printf( "%s\n",  str);
}
