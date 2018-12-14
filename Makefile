stencil: stencil.c
	mpicc -std=c99 -O3 -Wall $^ -o $@
clean:
	-rm -f stencil.out stencil stencil.pgm
