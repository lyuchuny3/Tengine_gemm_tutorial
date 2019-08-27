CC=gcc  -std=gnu99 -g 

EXE = test
objects = main.o tengine_4x16_kernel.o 

output:$(objects)
	$(CC) -o $(EXE) $(objects) -lopenblas
main.o:main.c
	$(CC) -c main.c -O3 

tengine_4x16_kernel.o:tengine_4x16_kernel.S
	$(CC) -c tengine_4x16_kernel.S

.PHONY: clean
clean:
	-rm $(EXE) $(objects)
