#include <string>
#include <iostream>

#define each(it, I) \
	for (typeof((I).begin()) it=(I).begin(); it!=(I).end(); ++it)

typedef enum comm_type {
	NOCOMM,
	DEFCOMM,
	NUM_COMMS
} comm_type;

using namespace std;

class SQL_command {
	public:
	comm_type type;		// "def", etc...
	
	public:
	SQL_command() {
		reset();
	}
	
	void reset() {
		type = NOCOMM;
	}
	
	void print(const int n = 0) {
		
	}
};

class SQL_DEF_command: public SQL_command {
	public:
	string name;
	string as;
	string from;
	
	SQL_DEF_command(string name_, string as_, string from_) {
		type = DEFCOMM;
		name = name_;
		as = as_;
		from = from_;
	}
	void reset() {
		as = "";
		from = "";
	}
};
