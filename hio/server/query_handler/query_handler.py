#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created by tianhao.wang at 9/27/18
    
"""

import abc

import sqlparse


class QueryHelper(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        pass

    # for the time being, query is written in a structured way
    def parse_query(self, query):
        parsed = sqlparse.parse(query)
        func = str(parsed[0].tokens[2].tokens[0])
        agg_column = str(parsed[0].tokens[2].tokens[1]).strip('()')
        where_clause = parsed[0].tokens[8]

        k = int((len(where_clause.tokens) - 3) / 4) + 1
        cols = []
        vals = []
        for i in range(k):
            pred = where_clause.tokens[2 + 4 * i]
            if type(pred.left) is sqlparse.sql.Identifier or type(pred.left) is sqlparse.sql.Operation:
                cols.append(str(pred.left))
                l = str(pred.right)
                vals.append((l, l))
            else:
                # range query, range query must follow the format of 17<=age<=25
                l = int(str(pred.left.left))
                cols.append(str(pred.left.right))
                r = int(str(pred.right))
                vals.append((l, r))
        # todo: for now, each query is one node; when `or' predicate is added, can be more
        nodes = [(cols, vals)]

        return [func, agg_column, nodes]
