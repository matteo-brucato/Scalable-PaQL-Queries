%{
#include <iostream>
#include <cstring>
#include <cctype>
#include <string>
#include <stdio.h>
#include <cmath>
#include "sql_query.hpp"
#include "sql_command.hpp"

#include "scanner.hpp"

#define YYSTYPE string

//~ #include "parser.hpp"


//#define DEBUG_ON
#ifdef DEBUG_ON
	#define debug(msg)			cerr << msg << endl;
#else
	#define debug(...)
#endif

using namespace std;

// Prototypes to keep the compiler happy
//void yyerror (const char *error);
int yylex();
void yyerror(const char *error) {
  cerr << error << endl;
} /* error handler */
//~ extern "C" {
    //~ extern int yylex();
//~ }

extern SQL_query q;
extern SQL_command c;
extern string ide;
%}

/* Require bison 2.3 or later */
%require "2.3"

/* add debug output code to generated parser. disable this for release
 * versions. */
%debug

//~ %pure-parser
//~ %expect 0
//~ %locations

//~ %parse-param {core_yyscan_t yyscanner}
//~ %lex-param   {core_yyscan_t yyscanner}


/* verbose error messages */
%error-verbose

%token IDE
//%token STRING
%token NUM
%token NAT_NUM
%token INFTY
//%token INTEGER
//%token EOL

%token SELECT
%token FROM
%token WHERE
%token PACKAGE
%token REPEAT
%token SUCH THAT
%token AS
%token ALL
%token DEF
%token LIMIT
%token ORDER GROUP BY
%token MAXIMIZE
%token MINIMIZE
%token INPUT
%token OUTPUT
%token DISTINCT

/* operators */
/* Precedence: lowest to highest */
//~ %nonassoc	SET				/* see relation_expr_opt_alias */
//~ %left		UNION EXCEPT
//~ %left		INTERSECT
%left		OR
%left		AND
%right		NOT
%right		'='
%nonassoc	'<' '>'
%left Op	/* multi-character ops */
//~ %left "<=" ">="
//~ %nonassoc	LIKE ILIKE SIMILAR
//~ %nonassoc	ESCAPE
//~ %nonassoc	OVERLAPS
%nonassoc	BETWEEN
%nonassoc	IN_P
//~ %left		POSTFIXOP		/* dummy for postfix Op rules */
/* Binary Operators*/
%left		'+' '-'
%left		'*' '/' '%'
%left		'^'
/* Unary Operators */
%left		AT				/* sets precedence for AT TIME ZONE */
%left		COLLATE
%right		UMINUS
%right      ALL
%left		'[' ']'
%left		'(' ')'
%left		TYPECAST
%left		'.'

%start start

%%

//input	: /* empty */
//		| line input
//;

start	//: 			{ return 3; }
		//: ';'			{ return 2; }
		
		: IDE			{ ide = $1;
						  return 1; }
		
		| query ';'		{ debug("Recognized Query..." << endl);
						  //q.print();
						  return 0; }
		
		| '#' garbage '#'
						{ debug("Recognized Comment..." << endl);
						  return 5; }
		
		| '#' query ';' '#'
						{ debug("Recognized Query Commented Out..." << endl);
						  return 5; }
		
		| command ';'	{ debug("Recognized Command..." << endl);
						  return 4; }
		
		| error ';'		{ yyerrok;
						  return -1; }
;

garbage	: IDE | IDE garbage;

command		: DEF IDE AS IDE FROM IDE		{ /*c = *new SQL_DEF_command($2, $4, $6);*/ }
			| DEF IDE AS NUM FROM IDE		{ /*c = *new SQL_DEF_command($2, $4, $6);*/ }
;

/* PACKAGE QUERY */
query	:   SELECT select_list
			FROM from_list
			where_clause
			such_that_clause
			group_by_clause
			order_by_clause
			objective_clause
			limit_clause
		{
			debug("{ PROJ: " << $2 << " }" << endl
				<< "{ RELS: " << $4 << " }" << endl
				<< "A query has been successfully parsed..." << endl << endl);
			//~ q.select.push_back($2);
			//~ q.from.push_back($4);

			q.json = "{\"SELECT\":[" + $2 + "]" + \
			         ",\"FROM\":[" + $4 + "]" + \
			         ",\"WHERE\":" + $5 + \
			         ",\"SUCH-THAT\":" + $6 + \
			         ",\"OBJECTIVE\":" + $9 + \
			         ",\"LIMIT\":" + $10 + \
			         "}";
			//cout << q.json << endl;
		}
;


/* COLUMN REFERENCE */
columnref	: IDE
				{
					$$ = "\"NODE_TYPE\":\"COL_REF\",\"attr_name\":\"" + $1 + "\"";
					//cout << "IDE" << endl;
					//~ $$ = makeColumnRef($1, NIL, @1, yyscanner);
				}
			| '*'
				{
					$$ = "\"NODE_TYPE\":\"COL_REF\",\"attr_name\":\"" + $1 + "\"";
				}
			| IDE indirection
				{
					$$ = "\"NODE_TYPE\":\"COL_REF\",\"table_name\":\"" + $1 + "\",\"attr_name\":\"" + $2 + "\"";
					//$$ = $1 + $2;
					//cout << "IDE indirection" << endl;
					//~ $$ = makeColumnRef($1, $2, @1, yyscanner);
				}
;
indirection	: indirection_el
			| indirection indirection_el
			| indirection '.' '*'
;
indirection_el	: '.' IDE
					{
						//cout << "POINT" << endl;
						$$ = $2;
						//~ $$ = (Node *) makeString($2);
					}
//				| '.' '*'
//					{
//						$$ = $2;
//						// $$ = (Node *) makeNode(A_Star);
//					}
;


/* RELATION REFERENCE */
relation	: IDE					{ $$ = "\"REL_NAME\":\"" + $1 + "\""; }
			| IDE '.' IDE			{ $$ = "\"REL_NAME\":\"" + $1 + "." + $3 + "\""; }
			| IDE IDE				{ $$ = "\"REL_NAME\":\"" + $1 + "\",\"REL_ALIAS\":\"" + $2 + "\""; }
			| IDE '.' IDE IDE		{ $$ = "\"REL_NAME\":\"" + $1 + "." + $3 + "\",\"REL_ALIAS\":\"" + $4 + "\""; }
			| '(' query ')' named	{ $$ = "";
									  debug("Nested query recognized..." + $4 << endl);
									  SQL_query* nested = new SQL_query;
									  *nested = q;
									  q.reset();
									  q.nested = nested; }
;

relation_repeat : relation REPEAT NAT_NUM   { $$ = $1 + ",\"REPEAT\":" + $3; }

named		:						{ $$ = "\"AS\":null"; }
			| IDE				    { $$ = "\"AS\":\"" + $1 + "\""; };
			| AS IDE				{ $$ = "\"AS\":\"" + $2 + "\""; };



/* SELECT CLAUSE */
//select_list	: attr_list;

select_list	: select_item								{ $$ = "{" + $1 + "}"; }
			| select_item ',' select_list				{ $$ = "{" + $1 + "}," + $3; }
;

attr_list	: attr_item								{ $$ = "{" + $1 + "}"; }
			| attr_item ',' attr_list				{ $$ = "{" + $1 + "}," + $3; }
;

// FIXME: It should be PACKAGE ( list of relation ALIASES, not NAMES )
select_item	: attr_item								{ $$ = $1; }
			| PACKAGE '(' relation_list ')' named	{ $$ = "\"NODE_TYPE\":\"PACKAGE\",\
															\"PACKAGE_RELS\":[" + $3 + "]," + \
															$5; }
;

attr_item	: columnref								{ $$ = $1; }
;



/* FROM CLAUSE */
from_list	: relation_list;

relation_list   : relation                                      { $$ = "{" + $1 + "}"; }
				| relation_repeat                               { $$ = "{" + $1 + "}"; }
				| relation ',' relation_list                    { $$ = "{" + $1 + "}," + $3; }
				| relation_repeat ',' relation_list             { $$ = "{" + $1 + "}," + $3; }
;



/* WHERE CLAUSE */
where_clause	: WHERE a_expr {
					debug("{ SELECTION: " << $2 << " }" << endl);
					$$ = $2;
				}
				| /* no where clause */ 	{ $$ = "null"; }



/* SUCH THAT CLAUSE */
such_that_clause	: SUCH THAT a_st_expr {
						debug("{ SUCH-THAT-CLAUSE: " << $2 << " }" << endl);
						$$ = $3;
					}
					| /* no such-that clause */ {
						$$ = "null";
					}
;

st_subquery :   SELECT a_st_expr
				FROM relation
				where_clause
				group_by_clause
			{
				$$ = "{\
				\"NODE_TYPE\":\"SUB_QUERY\",\
				\"CONTENT\":{\
				\"SELECT\":" + $2 + ",\
				\"FROM\":{" + $4 + "},\
				\"WHERE\":" + $5 + ",\
				\"GROUP BY\":" + $6 + \
				"}}";
			}
;

a_st_expr
	: c_st_expr
		{ $$ = $1; }

	| '(' a_st_expr ')'
		{ $$ = $2; }

	| ALL a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"ALL\",\"CONTENT\":" + $2 + "}"; }

	| '+' a_st_expr					%prec UMINUS
		{ $$ = "{\"NODE_TYPE\":\"UN_OP\",\"OP\":\"+\",\"left\":" + $2 + "}"; }
	| '-' a_st_expr					%prec UMINUS
		{ $$ = "{\"NODE_TYPE\":\"UN_OP\",\"OP\":\"-\",\"left\":" + $2 + "}"; }

	| a_st_expr '+' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"+\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr '-' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"-\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr '*' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"*\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr '/' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"/\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr '%' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"%\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr '^' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"^\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }

	| a_st_expr '<' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr '>' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\">\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr '=' a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"=\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr '<' '>' a_st_expr %prec Op
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<>\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
	| a_st_expr '!' '=' a_st_expr %prec Op
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"!=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
	| a_st_expr '<' '=' a_st_expr %prec Op
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
	| a_st_expr '>' '=' a_st_expr %prec Op
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\">=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
	| a_st_expr AND a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"AND\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| a_st_expr OR a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"OR\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| NOT a_st_expr
		{ $$ = "{\"NODE_TYPE\":\"UN_OP\",\"OP\":\"NOT\",\"left\":" + $2 + "}"; }
	| a_st_expr BETWEEN b_st_expr AND b_st_expr		%prec BETWEEN {
		$$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"AND\",\
			\"left\":{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\">=\",\"left\":" + $1 + ",\"right\":" + $3 + "},\
			\"right\":{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<=\",\"left\":" + $1 + ",\"right\":" + $5 + "}}";
		}
	| a_st_expr NOT BETWEEN b_st_expr AND b_st_expr	%prec BETWEEN {

		}
;

b_st_expr
	: c_st_expr
		{ $$ = $1; }
	| '+' b_st_expr					%prec UMINUS
		{ $$ = "{\"NODE_TYPE\":\"UN_OP\",\"OP\":\"+\",\"left\":" + $2 + "}"; }
	| '-' b_st_expr					%prec UMINUS
		{ $$ = "{\"NODE_TYPE\":\"UN_OP\",\"OP\":\"-\",\"left\":" + $2 + "}"; }
	| b_st_expr '<' b_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| b_st_expr '>' b_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\">\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| b_st_expr '=' b_st_expr
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"=\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
	| b_st_expr '!' '=' b_st_expr %prec Op
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"!=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
	| b_st_expr '<' '>' b_st_expr %prec Op
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<>\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
	| b_st_expr '<' '=' b_st_expr %prec Op
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
	| b_st_expr '>' '=' b_st_expr %prec Op
		{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\">=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }

/* Terminals for 2nd order expressions */
c_st_expr		: c_expr
				| func                              { $$ = "{" + $1 + "}"; }
				| '(' st_subquery ')'               { $$ = $2; }
;



/* OTHER CLAUSES */
group_by_clause	: GROUP BY attr_list				{ $$ = $3; }
				| /* No group by clause */          { $$ = "null"; }
;

order_by_clause	: ORDER BY attr_list				{ $$ = "order by " + $3; }
				| /* No order by clause */          { $$ = "null"; }
;

objective_clause	: MAXIMIZE c_st_expr			{ $$ = "{\"TYPE\":\"MAXIMIZE\",\"EXPR\":" + $2 + "}"; }
					| MINIMIZE c_st_expr			{ $$ = "{\"TYPE\":\"MINIMIZE\",\"EXPR\":" + $2 + "}"; }
					| /* No objective */			{ $$ = "null"; }
;

limit_clause	: LIMIT INPUT NAT_NUM				{ $$ = "{\"TYPE\":\"INPUT\",\"LIMIT\":" + $3 + "}"; }
				| LIMIT OUTPUT NAT_NUM				{ $$ = "{\"TYPE\":\"OUTPUT\",\"LIMIT\":" + $3 + "}"; }
				| /* No limit clause */				{ $$ = "null"; }
;

//~ func_name: IDE;
//~ func_args: func_arg_list | ;
//~ func_arg_list		: IDE
					//~ | NUM
					//~ | IDE ',' func_arg_list		{ $$ = $1 + "," + $3; }
					//~ | NUM ',' func_arg_list		{ $$ = $1 + "," + $3; }
//~ ;
func:
	func_name '(' func_arg_list ')' {
		$$ = "\"NODE_TYPE\":\"func\",\"func_name\":\"" + $1 + "\",\"func_args\":[" + $3 + "],\"distinct\":false";
	}
|	func_name '(' DISTINCT func_arg_list ')' {
		$$ = "\"NODE_TYPE\":\"func\",\"func_name\":\"" + $1 + "\",\"func_args\":[" + $4 + "],\"distinct\":true";
	}
//|	func_name '(' '*' ')' {
//		$$ = "{\"NODE_TYPE\":\"func\",\"func_name\":\"" + $1 + "\",\"func_args\":[\"*\"]}";
//	}
;

func_name:	IDE
;

func_arg_list	:func_arg_expr
					{
						$$ = $1;
					}
				| func_arg_list ',' func_arg_expr
					{
						$$ = $1 + "," + $3;
					}
;

func_arg_expr:  c_expr			{ $$ = $1; }
;



/*
 * General expressions
 * This is the heart of the expression syntax.
 *
 * We have two expression types: a_expr is the unrestricted kind, and
 * b_expr is a subset that must be used in some places to avoid shift/reduce
 * conflicts.  For example, we can't do BETWEEN as "BETWEEN a_expr AND a_expr"
 * because that use of AND conflicts with AND as a boolean operator.  So,
 * b_expr is used in BETWEEN and we remove boolean keywords from b_expr.
 *
 * Note that '(' a_expr ')' is a b_expr, so an unrestricted expression can
 * always be used by surrounding it with parens.
 *
 * c_expr is all the productions that are common to a_expr and b_expr;
 * it's factored out just to eliminate redundant coding.
 */
a_expr:		c_expr
				{ $$ = $1; }
		/*
		 * These operators must be called out explicitly in order to make use
		 * of bison's automatic operator-precedence handling.  All other
		 * operator names are handled by the generic productions using "Op",
		 * below; and all those operators will have the same precedence.
		 *
		 * If you add more explicitly-known operators, be sure to add them
		 * also to b_expr and to the MathOp list above.
		 */
			| '+' a_expr					%prec UMINUS
				{ $$ = "\"+\"" + $1; }
			| '-' a_expr					%prec UMINUS
				//~ { $$ = doNegate($2, @1); }
			| a_expr '+' a_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "+", $1, $3, @2); }
			| a_expr '-' a_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "-", $1, $3, @2); }
			| a_expr '*' a_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "*", $1, $3, @2); }
			| a_expr '/' a_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "/", $1, $3, @2); }
			| a_expr '%' a_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "%", $1, $3, @2); }
			| a_expr '^' a_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "^", $1, $3, @2); }
			| a_expr '<' a_expr
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
			| a_expr '>' a_expr
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\">\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
			| a_expr '=' a_expr
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"=\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
			| a_expr '!' '=' a_expr			%prec Op
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"!=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
			| a_expr '<' '>' a_expr			%prec Op
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<>\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
			| a_expr '<' '=' a_expr			%prec Op
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
			| a_expr '>' '=' a_expr			%prec Op
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\">=\",\"left\":" + $1 + ",\"right\":" + $4 + "}"; }
			| a_expr AND a_expr
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"AND\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
			| a_expr OR a_expr
				{ $$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"OR\",\"left\":" + $1 + ",\"right\":" + $3 + "}"; }
			| NOT a_expr
				{ $$ = "{\"NODE_TYPE\":\"UN_OP\",\"OP\":\"NOT\",\"left\":" + $2 + "}"; }
			/*
			 *	Ideally we would not use hard-wired operators below but
			 *	instead use opclasses.  However, mixed data types and other
			 *	issues make this difficult:
			 *	http://archives.postgresql.org/pgsql-hackers/2008-08/msg01142.php
			 */
			| a_expr BETWEEN b_expr AND b_expr		%prec BETWEEN {
				$$ = "{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"AND\",\
					\"left\":{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\">=\",\"left\":" + $1 + ",\"right\":" + $3 + "},\
					\"right\":{\"NODE_TYPE\":\"BIN_OP\",\"OP\":\"<=\",\"left\":" + $1 + ",\"right\":" + $5 + "}}";
				}
				//~ $$ = (Node *) makeA_Expr(AEXPR_AND, NIL,
					//~ (Node *) makeSimpleA_Expr(AEXPR_OP, ">=", $1, $4, @2),
					//~ (Node *) makeSimpleA_Expr(AEXPR_OP, "<=", $1, $6, @2),
										 //~ @2);
			| a_expr NOT BETWEEN b_expr AND b_expr	%prec BETWEEN
				{
					//~ $$ = (Node *) makeA_Expr(AEXPR_OR, NIL,
						//~ (Node *) makeSimpleA_Expr(AEXPR_OP, "<", $1, $5, @2),
						//~ (Node *) makeSimpleA_Expr(AEXPR_OP, ">", $1, $7, @2),
											 //~ @2);
				}
;

/*
 * Restricted expressions
 *
 * b_expr is a subset of the complete expression syntax defined by a_expr.
 *
 * Presently, AND, NOT, IS, and IN are the a_expr keywords that would
 * cause trouble in the places where b_expr is used.  For simplicity, we
 * just eliminate all the boolean-keyword-operator productions from b_expr.
 */
b_expr:		c_expr
				{ $$ = $1; }
			//~ | b_expr TYPECAST Typename
				//~ { $$ = makeTypeCast($1, $3, @2); }
			| '+' b_expr					%prec UMINUS
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "+", NULL, $2, @1); }
			| '-' b_expr					%prec UMINUS
				//~ { $$ = doNegate($2, @1); }
			| b_expr '+' b_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "+", $1, $3, @2); }
			| b_expr '-' b_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "-", $1, $3, @2); }
			| b_expr '*' b_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "*", $1, $3, @2); }
			| b_expr '/' b_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "/", $1, $3, @2); }
			| b_expr '%' b_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "%", $1, $3, @2); }
			| b_expr '^' b_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, "^", $1, $3, @2); }
			| b_expr '<' b_expr
				{ /*cout << "##################### <" << endl;*/ }
			| b_expr '>' b_expr
				//~ { $$ = (Node *) makeSimpleA_Expr(AEXPR_OP, ">", $1, $3, @2); }
			| b_expr '=' b_expr
			| b_expr '!' '=' b_expr			%prec Op
			| b_expr '<' '>' b_expr			%prec Op
			| b_expr '<' '=' b_expr			%prec Op
				{ /*cout << "##################### <=" << endl;*/ }
			| b_expr '>' '=' b_expr			%prec Op
;

/* Terminals for 1st order expressions */
c_expr		: columnref                 { $$ = "{" + $1 + "}";}
			| NUM
			| NAT_NUM
			| INFTY
			  { $$ = "\"infinity\""; }
			//~ | '(' a_expr ')' opt_indirection
;

//~ opt_indirection : /*EMPTY*/
				//~ | opt_indirection indirection_el
//~ ;





%%


/**** YYLEX ****/

bool is_integer(float k) {
  return std::floor(k) == k;
}

int yylex () {
	char c;
	
	// Ignore whitespaces, get first nonwhite character.
	//while ((c = getchar ()) == ' ' || c == '\t' || c == '\n');
	while (iswspace(c = getchar()));
	
	// Return when encouter end-of-file
	if (c == EOF) return 0;

	// Char starts a number => parse the number.
	if (c == '.' || isdigit (c)) {
		//cout << c << endl;
		char c2 = getchar();
		ungetc(c2, stdin);
		
		if (c != '.' || isdigit(c2)) {
			ungetc(c, stdin);
			
			double num;
			char *str = (char*) malloc(20);
			
			scanf("%lf", &num);
			sprintf(str, "%lf", num);
			yylval = (string) str;
			
			if (c2 != '.' && is_integer(num) && num >= 0) {
				//cout << "NUM_NAT: " << num << endl;
				//cout << "NUM_NAT: " << str << endl;
				scanf("%ld", (long int*)&num);
				sprintf(str, "%ld", (long int)num);
				yylval = (string) str;
				//cout << str << endl;
				return NAT_NUM;
			}
			
			return NUM;
		}
	}

	// Char starts an identifier => read the name.
	if (isalpha (c)) {
		string ide;
		
		do {
			ide += c;
			c = getchar();
		//} while (isalpha(c) || isdigit(c) || c == '_' || c == '.' || c == '<' || c == '>' || c == '=');
		} while (isalpha(c) || isdigit(c) || c == '_');
		ungetc(c, stdin);
		
		yylval = ide;
		
		if (strcasecmp(ide.c_str(), "INFINITY") == 0) return INFTY;
		if (strcasecmp(ide.c_str(), "SELECT") == 0) return SELECT;
		if (strcasecmp(ide.c_str(), "PACKAGE") == 0) return PACKAGE;
		if (strcasecmp(ide.c_str(), "REPEAT") == 0) return REPEAT;
		if (strcasecmp(ide.c_str(), "FROM") == 0) return FROM;
		if (strcasecmp(ide.c_str(), "WHERE") == 0) return WHERE;
		if (strcasecmp(ide.c_str(), "ORDER") == 0) return ORDER;
		if (strcasecmp(ide.c_str(), "GROUP") == 0) return GROUP;
		if (strcasecmp(ide.c_str(), "SUCH") == 0) return SUCH;
		if (strcasecmp(ide.c_str(), "THAT") == 0) return THAT;
		if (strcasecmp(ide.c_str(), "BY") == 0) return BY;
		if (strcasecmp(ide.c_str(), "AS") == 0) return AS;
		if (strcasecmp(ide.c_str(), "ALL") == 0) return ALL;
		if (strcasecmp(ide.c_str(), "LIMIT") == 0) return LIMIT;
		if (strcasecmp(ide.c_str(), "DEF") == 0) return DEF;
		if (strcasecmp(ide.c_str(), "AND") == 0) return AND;
		if (strcasecmp(ide.c_str(), "OR") == 0) return OR;
		if (strcasecmp(ide.c_str(), "NOT") == 0) return NOT;
		if (strcasecmp(ide.c_str(), "MAXIMIZE") == 0) return MAXIMIZE;
		if (strcasecmp(ide.c_str(), "MINIMIZE") == 0) return MINIMIZE;
		if (strcasecmp(ide.c_str(), "BETWEEN") == 0) return BETWEEN;
		if (strcasecmp(ide.c_str(), "INPUT") == 0) return INPUT;
		if (strcasecmp(ide.c_str(), "OUTPUT") == 0) return OUTPUT;
		if (strcasecmp(ide.c_str(), "DISTINCT") == 0) return DISTINCT;
		//~ if (strcasecmp(ide.c_str(), "<=") == 0) {cout << "LESS THAN" << endl; return LESS_THAN_OP;}
		//~ if (strcasecmp(ide.c_str(), ">=") == 0) return GREATER_THAN_OP;
		
		//cout << "IDE: " << yylval << endl;
		return IDE;
	}

	// Any other character is a token by itself.
	yylval = c;
	//cout << "OTHER TOKEN: " << c << endl;
	return c;
}

/*** MAIN ***
int main() {
	return yyparse();
}*/

//~ void yyerror(const char *error) { /* Called by yyparse on error */
	//~ cerr << error << endl;
//~ }
