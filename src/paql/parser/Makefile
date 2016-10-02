CC = g++
CFLAGS = -g -Wall
LDFLAGS = 

GRAMMAR_FILE = PaQL_Grammar.y

all: PaShell PaQL_Parser

#PaShell: PaShell.o parser.o /usr/local/lib/libjson_spirit.a
PaShell: PaShell.o parser.o
#	g++ -lfl -o $@ PaShell.o parser.o lexical.o /usr/local/lib/libjson_spirit.a
#PaServer: PaServer.o

#PaQL_Parser: parser.o PaQL_Parser.o /usr/local/lib/libjson_spirit.a
PaQL_Parser: parser.o PaQL_Parser.o

PaQL_Parser.o: PaQL_Parser.cpp
#lexical.o: lexical.cpp
parser.o: parser.cpp
PaShell.o: PaShell.cpp sql_query.hpp packets.hpp
#PaServer.o: PaServer.cpp packets.hpp

parser.cpp parser.dot: $(GRAMMAR_FILE) sql_query.hpp
	bison -d -o parser.cpp -g $(GRAMMAR_FILE)

#lexical.cpp: lexical.l parser.hpp
#	flex -o lexical.cpp lexical.l


parser-graph: parser.dot
	dot -Tpng parser.dot > parser.png

clean:
	rm -f parser.cpp parser.hpp lexical.cpp *.o *.dot parser.png PaShell PaServer PaQL_Parser
