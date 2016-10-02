import itertools
import math
from itertools import islice
from logging import warning
import operator
import chardet
from decimal import Decimal

import numpy as np
from mpmath import mpf



def iter_chunks(n, iterable):
    """
    Iterates iterable in "chunks" of size n
    """
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk



def int_round(x, base=1):
    assert base > 0
    return int(base * round(float(x) / base))



def str_u_gc(((aggr, attr), op, rhs)):
    return "{}({}) {} {}".format(aggr, attr, op_to_opstr(op), rhs)



def op_to_opstr(op):
    if op is operator.le:
        return "<="
    elif op is operator.ge:
        return ">="
    elif op is operator.eq:
        return "="
    elif op is operator.add:
        return "+"
    elif op is operator.sub:
        return "-"
    elif op is operator.mul:
        return "*"
    elif op is operator.div:
        return "/"
    else:
        raise Exception("Operator '{}' not supported yet.", op)



def opstr_to_op(opstr):
    if opstr == "<=":
        return operator.le
    elif opstr == ">=":
        return operator.ge
    elif opstr == "<":
        return operator.lt
    elif opstr == ">":
        return operator.gt
    elif opstr == "=":
        return operator.eq
    elif opstr == "-":
        return operator.sub
    elif opstr == "+":
        return operator.add
    elif opstr == "/":
        return operator.div
    elif opstr == "*":
        return operator.mul
    elif opstr == "and":
        return operator.and_
    elif opstr == "or":
        return operator.or_
    elif opstr == "not":
        return operator.not_
    else:
        raise Exception("Opstr not supported: {}".format(opstr))



def op_to_cplex_sense(op):
    if op == operator.le:
        return "L"
    elif op == operator.ge:
        return "G"
    elif op == operator.eq:
        return "E"
    else:
        raise Exception("Operator '{}' not supported yet.", op)



def binom_coeff(n, k):
    # return scipy.special.binom(n,k)
    try:
        return long(reduce(operator.mul, [(n + 1 - i) / i for i in xrange(1, k + 1)], 1.0))
    except OverflowError:
        return long(reduce(operator.mul, [(n + 1 - i) / i for i in xrange(1, k + 1)], mpf(1.0)))
    # return reduce(operator.mul, [n - i for i in xrange(k)], 1) / math.factorial(k)



def n_of_subsets(n, lb, ub):
    if lb is None or ub is None:
        raise Exception("Lower and upper bounds must be set.")
    # return 0
    # n_subsets = 0
    # for k in range(lb, ub + 1):
    # n_subsets += binom_coeff(n, k)
    assert lb >= 0 and ub <= n

    if lb == 0 and ub == n:
        return 2 ** n

    warning("ATTENTION! THIS FUNCTION MAY BE EXTREMELY EXPENSIVE!")

    if lb == 0:
        if ub <= n / 2:
            return sum(binom_coeff(n, k) for k in range(lb, ub + 1))
        else:
            return 2 ** n - sum(binom_coeff(n, k) for k in range(ub + 1, n + 1))
    if ub == n:
        if lb >= n / 2:
            return sum(binom_coeff(n, k) for k in range(lb, ub + 1))
        else:
            return 2 ** n - sum(binom_coeff(n, k) for k in range(0, lb))
    if lb <= n / 2 <= ub:
        return 2 ** n\
               - sum(binom_coeff(n, k) for k in range(0, lb))\
               - sum(binom_coeff(n, k) for k in range(ub + 1, n + 1))

    return sum(binom_coeff(n, k) for k in range(lb, ub + 1))



def pretty_table_str(rows, header=None, footers=None, formats=None, encodings=None, alignments=None):
    """
    :param rows: An iterable of either : (1) dictionaries or Tuples (for named table rows); (2) lists (nor unnamed
    rows).
    :param header: Which attributes to print out (if named).
    :return: A string.
    """
    from src.data_model.tuple import Tuple

    assert header is None or len(rows) == 0 or (isinstance(rows[0], dict) or isinstance(rows[0], Tuple))
    assert header is None or len(rows) == 0 or ((not isinstance(rows[0], dict) and not isinstance(rows[0], Tuple)) or
                                                all(set(header).issubset(set(row.iterkeys())) for row in rows))

    # Default formats
    use_formats = {
        float: ".6f",
        Decimal: ".6f",
        bool: "",
        int: "",
        long: "",
        str: "",
        unicode: "",
        list: "",
        type(None): "",
    }
    # print use_formats
    # Update defaults with user-specified values
    if formats is not None:
        use_formats.update(formats)

    # Default alignments
    use_alignment = {
        float: ">",
        Decimal: ">",
        bool: "<",
        int: ">",
        long: ">",
        str: "<",
        unicode: "<",
        list: "<",
        type(None): "<",
    }
    # Update defaults with user-specified values
    if alignments is not None:
        use_alignment.update(alignments)

    # Default encodings
    use_encodings = {
        float: lambda x: x,
        Decimal: lambda x: x,
        bool: lambda x: x,
        int: lambda x: x,
        long: lambda x: x,
        str: lambda x: char_decode(x).strip(),
        unicode: lambda x: char_decode(x).strip(),
        list: lambda x: char_decode(str(x)).strip(),
        type(None): lambda x: x,
    }
    # Update defaults with user-specified values
    if encodings is not None:
        use_encodings.update(encodings)

    def data_encode(x):
        return use_encodings[type(x)](x)

    def format_val(x, width=None):
        format_str = "{:" + \
                     use_alignment[type(x)] + \
                     (str(width) if width is not None else "") + \
                     use_formats[type(x)] + \
                     "}"
        return format_str.format(data_encode(x))

    # Get max lengths
    if header is not None and footers is not None:
        all_values = zip(header, *[ [ row[attr] for attr in header ] for row in rows ])
        for i in xrange(len(all_values)):
            for footer in footers:
                all_values[i] = list(all_values[i]) + [footer[i]]
    elif header is not None:
        all_values = zip(header, *[ [ row[attr] for attr in header ] for row in rows ])
    elif footers is not None:
        all_values = zip(*[ [ row[attr] for attr in header ] for row in rows ])
        for i in xrange(len(all_values)):
            for footer in footers:
                all_values[i] = list(all_values[i]) + [footer[i]]
    else:
        all_values = zip(*[ row for row in rows ])
    max_lens = [
        len(format_val(max(val, key=lambda x: len(format_val(x)))))
        for val in all_values
    ]

    ruler = "-+-".join("-" * m for m in max_lens)

    # Produce string
    string = [ruler]

    if header is not None:
        string.append(' | '.join(
            format_val(x, width=y)
            for x, y in zip(header, max_lens)))

        string.append(ruler)

    for row in rows:
        string.append(' | '.join(
            format_val(x, width=y)
            for x, y in zip([ row[attr] for attr in header ], max_lens)
        ))

    string.append(ruler)

    if footers is not None:
        for footer in footers:
            string.append(' | '.join(
                format_val(x, width=y)
                for x, y in zip(footer, max_lens)))

            string.append(ruler)

    string.append("({} rows)".format(len(rows)))

    return "\n".join(string)



def pretty_table_str_named(rows, headers=None):
    """
    Assumes the tuples are named tuple. Do not use with normal tuples.
    """
    if len(rows)==0:
        print rows
        return
    # raise

    # if len(rows) > 1:
    # print rows[0]._fields
    if headers is None:
        headers = rows[0]._fields

    # data = []
    # for row in rows:
    # for i, header in enumerate(headers):
    # 		datum = getattr(rows[0], header)
    # 		rows.append(datum)

    lens = []
    for i, header in enumerate(headers):
        lens.append(len(str(max([getattr(x, header) for x in rows] + [headers[i]], key=lambda x: len(str(x))))))
    # print lens

    # Generate formats
    hformats = []
    formats = []
    # data = []
    for i, header in enumerate(headers):
        datum = getattr(rows[0], header)
        # data.append(datum)
        if isinstance(datum, int):
            formats.append("%%%dd" % lens[i])
        else:
            formats.append("%%-%ds" % lens[i])
        hformats.append("%%-%ds" % lens[i])

    # Generate patterns
    hpattern = " | ".join(hformats)
    pattern = " | ".join(formats)

    # Generate result
    string = hpattern % tuple(headers) + "\n"
    string += "-+-".join(['-' * n for n in lens]) + "\n"
    for line in rows:
        # print pattern
        # print [ getattr(line, h) for h in headers]
        string += pattern % tuple(getattr(line, h) for h in headers) + "\n"
    return string



def str_result_table(rows):
    return '\n'.join([' '.join(["{:10}".format(repr(row[i])) for i in range(len(row))])
                      for row in rows])



def char_decode(s):
    assert isinstance(s, basestring)
    for i in ["utf-8", "ISO-8859-2"]:
        try:
            # Default is: encode("utf-8")
            return s.decode(i).encode()
        except Exception:
            pass

    try:
        # Default is: encode("utf-8")
        return s.decode(chardet.detect(s)["encoding"]).encode()
    except Exception:
        raise Exception("Cannot decode '{}'".format(s))



def avg(l):
    s, c = 0.0, 0
    for i in l:
        s += i
        c += 1
    return s / c



def avgp(l):
    return [np.mean(l, axis=0)]



def medianp(L):
    return [np.median(L, axis=0)]



def minp(L):
    return [np.min(L, axis=0)]



def maxp(L):
    return [np.max(L, axis=0)]



def randpoints(L, ratio):
    assert 0 <= ratio <= 1
    print "selecting {} random points from cluster containing {} tuples".format(math.ceil(len(L) * ratio), len(L))
    return [L[i] for i in np.random.choice(xrange(len(L)), math.ceil(len(L) * ratio), replace=False)]



def all_corners(L):
    # TODO: Change this to take the query into account and select only corners that matter for the query
    dims = len(L[0])

    # Compute min values and max values for each dimension
    all_mins = [None for d in xrange(dims)]
    all_maxs = [None for d in xrange(dims)]
    for l in L:
        for d in xrange(dims):
            if all_mins[d] is None or l[d] < all_mins[d]:
                all_mins[d] = l[d]
            if all_maxs[d] is None or l[d] > all_maxs[d]:
                all_maxs[d] = l[d]

    # Enumerate all possible corners
    for corner_funcs in itertools.product([min, max], repeat=dims):
        corner = []
        for d in xrange(dims):
            if corner_funcs[d]==min:
                corner.append(all_mins[d])
            elif corner_funcs[d]==max:
                corner.append(all_maxs[d])
            else:
                raise Exception
        yield corner



def from_feasible_sols(L, feasible_sols, all_attrs):
    for feasible_sol in feasible_sols:
        if feasible_sol is None: continue
        for feasible_sol_tuple in feasible_sol.iter_tuples():
            feasible_sol_tuple_t = tuple(getattr(feasible_sol_tuple, attr) for attr in all_attrs)
            if feasible_sol_tuple_t in L:
                print "INCLUDING TUPLE", feasible_sol_tuple_t
                yield feasible_sol_tuple_t



def iter_subsets(S, order, n_variations, include_empty=False):
    """
    Order "top-down" means from bigger to smaller elements. Order "bottom-up" means the opposite.
    """
    n = len(S) + 1
    if order=="top-down":
        for size in xrange(n + (1 if include_empty else 0)):
            for subset in itertools.islice(itertools.combinations(S, n - size), n_variations):
                yield subset
    elif order=="bottom-up":
        for size in xrange(0 if include_empty else 1, n):
            for subset in itertools.islice(itertools.combinations(S, size), n_variations):
                yield subset
    else:
        raise Exception("Order '{}' not recognized.".format(order))



def iter_supersets(S, U, order, n_variations, include_empty=False):
    """
    Emits all subsets of the universe U that are supersets of S.
    Order "top-down" means from bigger to smaller elements. Order "bottom-up" means the opposite.
    """
    # Missing elements in S
    M = [u for u in U if u not in S]
    # Generate all subsets of the missing elements and add them to S
    n = len(M) + 1
    if order=="top-down":
        for size in xrange(n + (1 if include_empty else 0)):
            for subset in itertools.islice(itertools.combinations(M, n - size), n_variations):
                yield S + list(subset)
    elif order=="bottom-up":
        for size in xrange(0 if include_empty else 1, n):
            for subset in itertools.islice(itertools.combinations(M, size), n_variations):
                yield S + list(subset)
    else:
        raise Exception("Order '{}' not recognized.".format(order))
