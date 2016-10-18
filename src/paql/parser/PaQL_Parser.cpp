#include <stdio.h>
#include <string.h>
#include <fstream>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <errno.h>
#include <iostream>
#include "sql_query.hpp"

using namespace std;

extern int yyparse();

/// This is filled up by the parser yyparse()
SQL_query q;
string ide;

string create_json_of_query() {
	return q.json;
}

int main(int argc, char *argv[]) {
	int c = getchar();
	
	// Remove initial empty spaces
	while (iswspace(c)) c = getchar();
	
	// Else: it's an input to be parsed, put the character back in place
	ungetc(c, stdin);
	
	// Parse query
	q.reset();
	int p = yyparse();
	
	// Switch result of yyparse()
	string JSON_q;
	int r;
	switch (p) {
		
		case -1: // Parser error
			cerr << argv[0] << " error: cannot parse query" << endl;
			break;
		
		case 1: // PaQL command (just an IDE, yet to be checked)
			cerr << argv[0] << " error: command `" << ide << "' not recognized" << endl;
			break;
		
		case 4: // SQL Command parsed correctly
			cerr << argv[0] << " error: cannot execute command" << endl;
			break;
		
		case 5: // Commented line
			//~ cout << endl;
			break;
		
		case 0: // SQL Query parsed correctly
			JSON_q = create_json_of_query();
			cout << JSON_q << endl;
			break;
		
		default:
			cerr << argv[0] << " error: parser returns unknown value" << endl;
	}
	
	return 0;
}
