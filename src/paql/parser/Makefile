CC = g++
CFLAGS = -g -Wall
LDFLAGS = 

GRAMMAR_FILE = PaQL_Grammar.y

all: PaQL_Parser

PaQL_Parser: parser.o PaQL_Parser.o

PaQL_Parser.o: PaQL_Parser.cpp
parser.o: parser.cpp

parser.cpp parser.dot: $(GRAMMAR_FILE) sql_query.hpp
	bison -d -o parser.cpp -g $(GRAMMAR_FILE)

parser-graph: parser.dot
	dot -Tpng parser.dot > parser.png

clean:
	rm -f parser.cpp parser.hpp lexical.cpp *.o *.dot parser.png PaQL_Parser
