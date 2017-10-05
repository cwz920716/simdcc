import numpy as np
import abc
from abc import ABCMeta
from itertools import zip_longest
import contextlib
from enum import Enum
import operator
import sys
import copy

import sympy
from sympy import symbols, sympify
from sympy.tensor import Indexed, IndexedBase, Idx

import islpy as isl

from dbtools import debug
import cgraph
from cgraph import Call, Inspect, Begin, If, SyntacticLoop

def simplify(domain, context):
    return domain.gist(context)

def project(domain, level, keep_dim=True):
    levels = domain.dim(isl.dim_type.set)
    out_names = {}
    out_levels = levels - (level + 1)
    for i in range(level+1, levels):
        out_names[i] = domain.get_dim_name(isl.dim_type.set, i)
    P = domain.project_out(isl.dim_type.set, level+1, out_levels)
    if keep_dim:
        P = P.add_dims(isl.dim_type.set, out_levels)
        for k, v in out_names.items:
            P = P.set_dim_name(isl.dim_type.set, k, v)
    return P

I = cgraph.IndUtils.axis2ind(0)
J = cgraph.IndUtils.axis2ind(1)
K = cgraph.IndUtils.axis2ind(2)
W = cgraph.IndUtils.axis2ind(3)

def S(idx, inds):
    name = 'S' + str(idx)
    Si = IndexedBase(name)
    return Si[inds]

class Statement(object):
    def __init__(self, inds, domain, sym, schedule=None, body=None):
        self.inds = inds
        self.domain = domain          # Polyhedera domain
        self.sym = sym
        assert len(self.sym.indices) == len(self.inds)
        for i in range(len(self.inds)):
            assert self.sym.indices[i] == self.inds[i]
        self.schedule = schedule      # scattering functions, if None then use inds
        self.body = body
        if self.schedule is None:
            self.schedule = inds

    def broadcast_schedule(ndim):
        if ndim > len(self.schedule):
            for i in range(0, ndim - len(self.schedule)):
                self.schedule.append(0)
        return

    @property
    def level(self):
        return self.schedule_domain.n_dim

    @property
    def schedule_as_isl_map(self):
        in_set = ', '.join([ str(x) for x in self.inds ])
        out_set = ', '.join([ str(x) for x in self.schedule ])
        desc = '{ ' + '[' + in_set + '] -> [' + out_set + ']' + ' }'
        m = isl.BasicMap(desc)
        debug('schedule: ' + str(m))
        return m

    @property
    def schedule_as_isl_named_map(self):
        in_set = ', '.join([ str(x) for x in self.inds ])
        out_set = ', '.join([ str(x) for x in self.schedule ])
        desc = '{ ' + str(self.sym.base) + '[' + in_set + '] -> ' + str(self.sym.base)  + '[' + out_set + ']' + ' }'
        m = isl.BasicMap(desc)
        debug('schedule: ' + str(m))
        return m

    @property
    def domain_as_isl_set(self):
        dims = ', '.join([ str(x) for x in self.inds ])
        desc = '{ ' + '[' + dims + ']' + ': ' + self.domain + ' }'
        s = isl.BasicSet(desc)
        debug('domain: ' + str(s))
        return s

    @property
    def domain_as_isl_named_set(self):
        dims = ', '.join([ str(x) for x in self.inds ])
        desc = '{ ' + str(self.sym.base) + '[' + dims + ']' + ': ' + self.domain + ' }'
        s = isl.BasicSet(desc)
        debug('domain: ' + str(s))
        return s

    @property
    def named_domain_string(self):
        dims = ', '.join([ str(x) for x in self.inds ])
        desc = str(self.sym.base) + '[' + dims + ']' + ': ' + self.domain
        return desc

    @property
    def semi_named_schedule_string(self):
        in_set = ', '.join([ str(x) for x in self.inds ])
        out_set = ', '.join([ str(x) for x in self.schedule ])
        desc = str(self.sym.base) + '[' + in_set + '] -> ' + 'T[' + out_set + ']'
        return desc
        
    @property
    def schedule_domain(self):
        return self.domain_as_isl_set.apply(self.schedule_as_isl_map)

    def dump_isl_code(self):
        ctx = isl.BasicSet('{ : }')
        dom = self.domain_as_isl_named_set
        sched = self.schedule_as_isl_named_map
        sd = sched.intersect_domain(dom)
        debug(sd)
        build = isl.AstBuild.from_context(ctx)
        ast = build.node_from_schedule_map(sd)
        def printASTAsC(ast):
            p = isl.Printer.to_str(ast.get_ctx())
            p = p.set_output_format(isl.format.C)
            p = p.print_ast_node(ast)
            print(p.get_str())
        printASTAsC(ast)

    @property
    def isl_ast(self):
        ctx = isl.BasicSet('{ : }')
        dom = self.domain_as_isl_named_set
        sched = self.schedule_as_isl_named_map
        sd = sched.intersect_domain(dom)
        build = isl.AstBuild.from_context(ctx)
        return build.node_from_schedule_map(sd)

    def codegen(self, printer):
        abs_body = Inspect(self.sym)
        # TODO: Implement the algorithm from paper "Polyhedral AST generation is more than scanning polyhedra"
        loop = SyntacticLoop(0, [], [], abs_body)
        return loop.codegen(printer)

    def isl_codegen(self, printer):
        ast = self.isl_ast
        debug(ast)
        k = str(self.sym.base)
        expr_map = {}
        expr_map[k] = self.body
        sym_map = {}
        sym_map[k] = self.sym
        ast_proc = AstProcessor(sym_map, expr_map)
        ast_proc.convert(ast).codegen(printer)
        return

class MultiStatements(object):
    def __init__(self, statements):
        self.statements = statements

    # MultiStatement *MUST* have different names for each statement
    @property
    def domain_as_isl_named_set(self):
        desc = '{' + ';'.join( [s.named_domain_string for s in self.statements] ) + '}'
        s = isl.UnionSet(desc)
        debug(s)
        return s

    @property
    def schedule_as_isl_named_map(self):
        desc = '{' + ';'.join( [s.semi_named_schedule_string for s in self.statements] ) + '}'
        s = isl.UnionMap(desc)
        debug(s)
        return s

    def schedule_ndim(self):
        slist = [ len(s.schedule) for s in self.statements ]
        return max(slist)

    @property
    def isl_ast(self):
        pass

    def dump_isl_code(self):
        ctx = isl.BasicSet('{ : }')
        dom = self.domain_as_isl_named_set
        sched = self.schedule_as_isl_named_map
        sd = sched.intersect_domain(dom)
        debug(sd)
        build = isl.AstBuild.from_context(ctx)
        ast = build.node_from_schedule_map(sd)
        def printASTAsC(ast):
            p = isl.Printer.to_str(ast.get_ctx())
            p = p.set_output_format(isl.format.C)
            p = p.print_ast_node(ast)
            print(p.get_str())
        printASTAsC(ast)

    @property
    def isl_ast(self):
        ctx = isl.BasicSet('{ : }')
        dom = self.domain_as_isl_named_set
        sched = self.schedule_as_isl_named_map
        sd = sched.intersect_domain(dom)
        build = isl.AstBuild.from_context(ctx)
        return build.node_from_schedule_map(sd)

    def codegen(self, printer):
        pass

    def isl_codegen(self, printer):
        ast = self.isl_ast
        debug(ast)
        expr_map = {}
        sym_map = {}
        for S in self.statements:
            k = str(S.sym.base)
            expr_map[k] = S.body
            sym_map[k] = S.sym
        ast_proc = AstProcessor(sym_map, expr_map)
        ast_proc.convert(ast).codegen(printer)
        return


class AstProcessor(object):
    def __init__(self, sym_map, expr_map):
        self.sym_map = sym_map
        self.expr_map = expr_map
        return

    def convert(self, cmd):
        if cmd is None:
            return None
        fname = 'convert_%s' % type(cmd).__name__
        try:
            fn = getattr(self, fname)
        except AttributeError:
            raise NotImplementedError(fname)
        else:
            try:
                return fn(cmd)
            except:
                raise

    def convert_AstNode(self, ast):
        ast_type = ast.get_type()
        if ast_type == isl.ast_node_type.for_:
            return self.convert_for(ast)
        elif ast_type == isl.ast_node_type.if_:
            return self.convert_if(ast)
        elif ast_type == isl.ast_node_type.block:
            return self.convert_block(ast)
        elif ast_type == isl.ast_node_type.mark:
            return self.convert_mark(ast)
        elif ast_type == isl.ast_node_type.user:
            return self.convert(ast.user_get_expr())

    def convert_block(self, ast):
        children = ast.block_get_children()
        children_cmd = []
        for i in range(children.n_ast_node()):
            x = children.get_ast_node(i)
            children_cmd.append(self.convert(x))
        return Begin(children_cmd)

    def convert_AstExpr(self, ast):
        ast_type = ast.get_type();
        if ast_type == isl.ast_expr_type.op:
            debug(ast.to_C_str())
            op = ast.get_op_type()
            if op == isl.ast_op_type.call:
                # process S1(i, j, ...)
                n_args = ast.get_op_n_arg()
                call = ast.get_op_arg(0)
                assert call.get_type() == isl.ast_expr_type.id
                Sx = call.get_id().get_name()
                real_statement = copy.copy(self.expr_map[Sx])
                if real_statement is None:
                    return Call(Sx, [self.convert(ast.get_op_arg(i+1)) for i in range(n_args-1)])
                for i in range(n_args - 1):
                    arg_idx = i + 1
                    to_expr = self.convert(ast.get_op_arg(arg_idx))
                    from_expr = self.sym_map[Sx].indices[i]
                    real_statement.replace(from_expr, to_expr)
                return real_statement
            else:
                expr_str = ast.to_C_str()
                return sympify(expr_str)
        elif ast_type == isl.ast_expr_type.id:
            ind_string = ast.get_id().get_name()
            return symbols(ind_string)
        elif ast_type == isl.ast_expr_type.int:
            inc = ast.get_val().to_python()
            return inc
        else:
            raise NotImplementedError(ast)

    """
       return (UpperBound, Inclusive) implies ast is a </<= Ast
    """
    def get_upper_bound(self, ast):
        assert ast.get_type() == isl.ast_expr_type.op
        op_type = ast.get_op_type()
        if op_type == isl.ast_op_type.lt:
            assert ast.get_op_n_arg() == 2
            ub = self.convert(ast.get_op_arg(1))
            return (ub, False)
        elif op_type == isl.ast_op_type.le:
            assert ast.get_op_n_arg() == 2
            ub = self.convert(ast.get_op_arg(1))
            return (ub, True)
        else:
            raise NotImplementedError(op_type)

    def convert_if(self, ast):
        cond = ast.if_get_cond()
        then_ = ast.if_get_then()
        else_ = ast.if_get_else if ast.if_has_else() else None
        return If(self.convert(cond), self.convert(then_), self.convert(else_))

    def convert_for(self, ast):
        it = ast.for_get_iterator()
        init = ast.for_get_init()
        cond = ast.for_get_cond()
        inc = ast.for_get_inc()
        body = ast.for_get_body()
        it = self.convert(it)
        inc = self.convert(inc)
        assert inc >= 0
        init = self.convert(init)
        ub, inclusive = self.get_upper_bound(cond)
        body = self.convert(body)
        return SyntacticLoop(1, [it], [slice(init, ub, inc)], body, inclusive=inclusive)
