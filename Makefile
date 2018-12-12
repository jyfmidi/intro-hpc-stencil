stencil: stencil.c
	gcc -std=c99 -O3 -march=native -mtune=native -Wall $^ -o $@
clean:
	-rm -f stencil.out stencil stencil.pgm
