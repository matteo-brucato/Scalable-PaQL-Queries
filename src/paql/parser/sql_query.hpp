#include <list>
#include <string>
#include <iostream>

#define each(it, I) \
	for (typeof((I).begin()) it=(I).begin(); it!=(I).end(); ++it)

using namespace std;

class Cond {};

typedef struct func {
	string name;
	list<string> args;
} func;

class SQL_query {
public:
	string json;
	
	list<string> select;	// List of columns
	list<string> from;		// List of relations or empty if nested
	SQL_query* nested;		// Used for nested query
	list<Cond> where;		// List of conditions in conjunctive form
	list<string> groupBy;
	list<string> orderBy;
	list<string> maximize;
	int stopAfter;
	string hint;
	
	SQL_query() {
		nested = NULL;
		reset();
	}
	
	void reset() {
		/** @todo Add recursive deletion */
		//if (nested != NULL) nested->reset();
		if (nested != NULL) delete nested;
		nested = NULL;
		select.clear();
		from.clear();
		where.clear();
		groupBy.clear();
		orderBy.clear();
		maximize.clear();
		stopAfter = -1;
		hint = "";
	}
	
	void print(const int n = 0) {
		string s;
		
		for (int i=0; i<n*4; i++) cout << " ";
		cout << "PROJECTIONS:\t";
		each(s, select) {
			cout << *s << ",";
		} cout << endl;
		
		for (int i=0; i<n*4; i++) cout << " ";
		cout << "RELATIONS:\t";
		if (nested == NULL) {
			each(s, from) {
				cout << *s << ",";
			}
		}
		else {
			cout << "(" << endl;
			nested->print(n+1);
			for (int i=0; i<(n+1)*4; i++) cout << " ";
			cout << ")";
		}
		cout << endl;
		
		if (groupBy.size()) {
			for (int i=0; i<n*4; i++) cout << " ";
			cout << "GROUP BY ";
			each(s, groupBy) {
				cout << *s << ",";
			} cout << endl;
		}
		if (orderBy.size()) {
			for (int i=0; i<n*4; i++) cout << " ";
			cout << "ORDER BY ";
			each(s, orderBy) {
				cout << *s << ",";
			} cout << endl;
		}
		if (stopAfter >= 0) {
			for (int i=0; i<n*4; i++) cout << " ";
			cout << "STOP AFTER " << stopAfter << endl;
		}
		if (maximize.size()) {
			for (int i=0; i<n*4; i++) cout << " ";
			cout << "MAXIMIZE ";
			each(s, maximize) {
				cout << *s << ",";
			} cout << endl;
		}
		if (hint != "") {
			for (int i=0; i<n*4; i++) cout << " ";
			cout << "HINT " << hint << endl;
		}
	}
};
