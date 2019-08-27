CC=gcc  -std=gnu99 -g 

EXE = test
objects = main.o

output:$(objects)
	$(CC) -o $(EXE) $(objects)  
main.o:main.c
	$(CC) -c main.c -O3 

.PHONY: clean
clean:
	-rm $(EXE) $(objects)
