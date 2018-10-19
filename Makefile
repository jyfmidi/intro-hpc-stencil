stencil: stencil.c
	gcc -std=c99 -O3 -march=native -mtune=native -ffast-math -Wall $^ -o $@

