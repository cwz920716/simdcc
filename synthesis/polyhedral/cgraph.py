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
from sympy import symbols
from sympy.tensor import Indexed, IndexedBase, Idx

import islpy

from dbtools import debug

class _Symbol(metaclass=ABCMeta):
    pass

"""
   A Tensor Symbol represents an N-d array that requires physical storage
   The type of N-d array is (shape, dtype) pair
   static typing requires the shape of every tensor, view, expression can be inferred statically
"""
class Tensor(_Symbol):
    def __init__(self, shape, dtype, name='tensor', value=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.value = value

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_scalar(self):
        return len(self.shape) == 0

    @property
    def is_const(self):
        return value is not None

    def __add__(self, rhs):
        return Add([ self, rhs ])

    def lower_sub(self, inds):
        T = IndexedBase(self.name)
        return T[inds]

    def lower(self):
        return self

    def lower_ref(self):
        return self.name

def MatrixXf(n1, n2, name):
    return Tensor((n1, n2), np.dtype('float32'), name=name)

def MatrixXd(n1, n2, name):
    return Tensor((n1, n2), np.dtype('float64'), name=name)

def Double(val, name='Scalar'):
    return Tensor((), np.dtype('float64'), name=name, value=val )

"""
   A View Symbol represents an view of asome N-d array, i.e., an N-d array which borrows storage fron another
"""
class View(_Symbol):
    def __init__(self, ref, slices, dtype, name='view'):
        self.ref = ref
        self.slices = slices
        self.dtype = dtype
        self.name = name

    @property
    def is_const(self):
        return ref.is_const

class Expr(_Symbol):
    def __init__(self, expr, name='rvalue'):
        self.expr = expr
        self.name = name

    @property
    def is_const(self):
        return True

"""
   All supported operator:
      SetItem
      GetItem
      Save
      Copy2 --Root
      Broadcast
      Conv1D
      Collapse
      ===========
      Reshape
      Elementwise
      Reduce
      Dot
      Stencil
      Conv2D
      Pad
      Select
      Slice
      Concat(axis, ...)
      Transpose
      Sequential --Root
   Macros:
      WHILE_LOOP
      AUTO_GRADIENT
"""
class Operator(metaclass=ABCMeta):
    """
       Tensor inputs for data flow analysis
    """
    @abc.abstractmethod
    def inputs(self):
        return None

    @abc.abstractproperty
    def shape(self):
        return None

    @abc.abstractproperty
    def dtype(self):
        return None

    @property
    def need_workspace(self):
        return False

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def n_inputs(self):
        return len(self.inputs)

    def __add__(self, rhs):
        return Add([ self, rhs ])

    def lower_ref(self):
        if self.need_workspace:
            return self.workspace.lower_ref()
        else:
            raise NotImplementedError("lowe_ref() invoked on operator without workspace!")

def broadcast_shapes(*shapes):
    """
    Determines output shape from broadcasting arrays.
    Parameters
    ----------
    shapes : tuples
        The shapes of the arguments.
    Returns
    -------
    output_shape : tuple
    Raises
    ------
    ValueError
        If the input shapes cannot be successfully broadcast together.
    """
    if len(shapes) == 1:
        return shapes[0]
    out = []
    for sizes in zip_longest(*map(reversed, shapes), fillvalue=-1):
        dim = max(sizes)
        if any(i != -1 and i != 1 and i != dim and not np.isnan(i) for i in sizes):
            raise ValueError("operands could not be broadcast together with "
                             "shapes {0}".format(' '.join(map(str, shapes))))
        out.append(dim)
    return tuple(reversed(out))

class SetItem(Operator):
    def __init__(self, dst, index, src):
        self.dst = dst
        self.index = index
        self.src = src

    @property
    def inputs(self):
        return [self.dst]

    @property
    def dtype(self):
        return self.dst.dtype

    @property
    def shape(self):
        return ()

    def lower(self):
        return Command( Opcode.ASSIGN, [ self.dst.lower_sub([ x for x in self.index ]), self.src ] )   

class GetItem(Operator):
    def __init__(self, src, index):
        self.src = src
        self.index = index

    @property
    def inputs(self):
        return [self.src]

    @property
    def dtype(self):
        return self.src.dtype

    @property
    def shape(self):
        return ()

    def lower(self):
        return self.src.lower_sub([ x for x in self.index ])

class Copy2(Operator):
    def __init__(self, dst, src):
        assert isinstance(dst, Tensor)
        self.dst = dst
        self.src = src

    @property
    def inputs(self):
        return [self.dst, self.src]

    @property
    def dtype(self):
        return self.dst.dtype

    @property
    def shape(self):
        return self.dst.shape

    def lower(self):
        inds = [ IndUtils.axis2ind(i) for i in range(self.ndim) ]
        slices = [ slice( 0, self.shape[i], 1 ) for i in range(self.ndim) ]
        body = Command( Opcode.ASSIGN, [self.dst.lower_sub(inds), self.src.lower_sub(inds)] )
        fl = SyntacticLoop(self.ndim, inds, slices, body)
        return fl

class Save(Operator):
    def __init__(self, src, name):
        self.src = src
        self.name = name
        self.workspace = Tensor(self.shape, self.dtype, name=self.name)

    @property
    def inputs(self):
        return [ self.src ]

    @property
    def dtype(self):
        return self.src.dtype

    @property
    def shape(self):
        return self.src.shape

    @property
    def need_workspace(self):
        return True

    def lower_sub(self, inds):
        return self.workspace.lower_sub(inds)

    def lower(self):
        inds = [ IndUtils.axis2ind(i) for i in range(self.ndim) ]
        slices = [ slice( 0, self.shape[i], 1 ) for i in range(self.ndim) ]
        body = Command( Opcode.ASSIGN, [self.workspace.lower_sub(inds), self.src.lower_sub(inds)] )
        fl = SyntacticLoop(self.ndim, inds, slices, body)
        return fl

class Broadcast(Operator):
    def __init__(self, src, shape):
        self.src = src
        self.broadcasted_shape = broadcast_shapes(src.shape, shape)

    @property
    def inputs(self):
        return [self.src]

    @property
    def dtype(self):
        return self.src.dtype

    @property
    def shape(self):
        return self.broadcasted_shape

    def lower_sub(self, inds):
        assert len(inds) == len(self.shape)
        offset = len(inds) - len(self.src.shape)
        assert offset >= 0
        src_ndim = self.src.ndim
        true_inds = [ 0 if self.src.shape[i] == 1 else inds[i + offset] for i in range(src_ndim) ]
        return self.src.lower_sub(true_inds)

class Conv1D(Operator):
    def __init__(self, src, axis, filter_weights={}):
        self.src = src
        self.axis = axis
        self.filter_weights = filter_weights

    @property
    def inputs(self):
        return [self.src]

    @property
    def dtype(self):
        return self.src.dtype

    @property
    def shape(self):
        return self.src.shape

    def lower_sub(self, inds):
        operands = 0
        for offset, weight in self.filter_weights.items():
            filter_inds = inds.copy()
            filter_inds[self.axis] = inds[self.axis] + offset
            operands += weight * self.src.lower_sub(filter_inds)
        return operands

def get_strides_from_shape(shape):
    s = 1
    ss = []
    for i in reversed(shape):
        ss.append(s)
        s *= i
    return tuple(reversed(ss))

class Collapse(Operator):
    def __init__(self, src, dims):
        self.src = src
        self.dims = dims
        self.dim_strides = get_strides_from_shape([ self.src.shape[i] for i in dims ])
        debug('shape=' + str(self.shape))

    @property
    def inputs(self):
        return [self.src]

    @property
    def dtype(self):
        return self.src.dtype

    @property
    def shape(self):
        new_shape = []
        dim_start = self.dims[0]
        dim_end = self.dims[-1] + 1
        for i in range(0, dim_start):
            new_shape.append(self.src.shape[i])
        prod = 1
        for i in range(dim_start, dim_end):
           prod *= self.src.shape[i]
        new_shape.append(prod)
        for i in range(dim_end, self.src.ndim):
            new_shape.append(self.src.shape[i])
        return tuple(new_shape)

    def lower_sub(self, inds):
        iter_inds = 0
        src_inds = []
        dim_start = self.dims[0]
        dim_end = self.dims[-1] + 1
        for i in range(0, dim_start):
            src_inds.append(inds[iter_inds])
            iter_inds += 1
        prod_inds = inds[iter_inds]
        for s in self.dim_strides:
            src_inds.append(prod_inds / s)
            prod_inds = prod_inds % s
        iter_inds += 1
        for i in range(dim_end, self.src.ndim):
            src_inds.append(inds[iter_inds])
            iter_inds += 1
        return self.src.lower_sub(src_inds)

class Concat(Operator):
    def __init__(self, sources, dim, name):
        assert len(sources) > 1
        self.sources = sources
        self.dim = dim
        self.name = name
        self.bounds = np.cumsum([ x.shape[dim] for x in sources ], dtype=np.dtype('int32'))
        self.bounds = [ int(x) for x in self.bounds ]
        self.workspace = Tensor(self.shape, self.dtype, name=self.name)

    @property
    def inputs(self):
        return self.sources

    @property
    def dtype(self):
        return self.sources[0].dtype

    @property
    def shape(self):
        head = self.sources[0]
        dst_shape = [ s for s in head.shape ]
        dst_shape[self.dim] = sum([ x.shape[self.dim] for x in self.sources ])
        return tuple(dst_shape)

    @property
    def need_workspace(self):
        return True

    def lower_sub(self, inds):
        return self.workspace.lower_sub(inds)

    def lower(self):
        inds = [ IndUtils.axis2ind(i) for i in range(self.ndim) ]
        slices = [ slice( 0, self.shape[i], 1 ) for i in range(self.ndim) ]
        concat_ind = inds[self.dim]
        args = []
        for i in range(0, self.n_inputs):
            cond = (concat_ind < self.bounds[i])
            inds_cond = copy.copy(inds)
            inds_cond[self.dim] = concat_ind - (self.bounds[i - 1] if i >= 1 else 0)
            stmt_cond = Command( Opcode.ASSIGN, [self.workspace.lower_sub(inds), self.sources[i].lower_sub(inds_cond)] )
            args.append(cond)
            args.append(stmt_cond)
        body = Command(Opcode.COND, args)
        fl = SyntacticLoop(self.ndim, inds, slices, body)
        return fl

class Dot(Operator):
    def __init__(self, lhs, rhs, name, transa=False, transb=False):
        self.lhs = lhs
        self.rhs = rhs
        self.name = name
        self.workspace = Tensor(self.shape, self.dtype, name=self.name)
        self.transa = transa
        self.transb = transb
        assert len(lhs.shape) == 2
        assert len(rhs.shape) == 2
        assert lhs.shape[1] == rhs.shape[0]
        assert lhs.dtype == rhs.dtype

    @property
    def inputs(self):
        return [self.lhs, self.rhs]

    @property
    def dtype(self):
        return self.lhs.dtype

    @property
    def shape(self):
        return (self.lhs.shape[0], self.rhs.shape[1])

    @property
    def need_workspace(self):
        return true

    def lower_sub(self, inds):
        return self.workspace.lower_sub(inds)

    def lower(self):
        # FIXME: BLAS call or naive matrix multiplication?
        dgemm = Command(Opcode.CALL, ['cblas_dgemm', [self.lhs.lower_ref(), self.rhs.lower_ref()]])
        return Command(Opcode.ASSIGN, [self.workspace.lower_ref(), dgemm])

class universal_func(object):
    def __init__(self, fn, n_inputs=-1, overload=False, op=None):
        self.fn = fn
        self.n_inputs = n_inputs
        self.overload = overload
        self.op = op

    def lower(self, operands):
        if self.n_inputs > 0:
            assert self.n_inputs == len(operands)
        if not self.overload:
            return Command(Opcode.CALL, [self.fn, operands])
        else:
            return self.op(*operands)
ufunc_add = universal_func('Add', n_inputs=2, overload=True, op=operator.add)
ufunc_sub = universal_func('Sub', n_inputs=2, overload=True, op=operator.sub)
ufunc_mul = universal_func('Mul', n_inputs=2, overload=True, op=operator.mul)
ufunc_div = universal_func('Div', n_inputs=2, overload=True, op=operator.truediv)
ufunc_pow = universal_func('Power', n_inputs=2)

class ElementWise(Operator):
    def __init__(self, sources, func):
        assert len(sources) > 0
        self.sources = sources
        self.src1 = self.sources[0]
        self.func = func

    @property
    def inputs(self):
        return self.sources

    @property
    def dtype(self):
        return self.src1.dtype

    @property
    def shape(self):
        return self.src1.shape

    def lower_sub(self, inds):
        return self.func.lower([ x.lower_sub(inds) for x in self.sources ])

class Add(ElementWise):
    def __init__(self, sources):
        super(Add, self).__init__(sources, ufunc_add)

class Sub(ElementWise):
    def __init__(self, sources):
        super(Sub, self).__init__(sources, ufunc_sub)

class Mul(ElementWise):
    def __init__(self, sources):
        super(Mul, self).__init__(sources, ufunc_mul)

class Div(ElementWise):
    def __init__(self, sources):
        super(Div, self).__init__(sources, ufunc_div)

class Pow(ElementWise):
    def __init__(self, sources):
        super(Pow, self).__init__(sources, ufunc_pow)

def define_Tensor(T):
    shape_string = ", ".join([ str(x) for x in T.shape ])
    if T.ndim == 2 and T.dtype == np.dtype('float32'):
         class_string = 'MatrixXf'
    elif T.ndim == 2 and T.dtype == np.dtype('float64'):
         class_string = 'MatrixXd'
    else:
         class_string = 'Tensor<' + str(T.dtype) + ', ' + T.ndim + '>'
    name_string = T.name
    return class_string + ' ' + name_string + '(' + shape_string + ');'

class IndUtils:
    desc = ['i', 'j', 'k', 'w']

    @classmethod
    def axis2ind(cls, i):
        if i < len(cls.desc):
            return symbols(cls.desc[i], integer=True)
        else:
            return symbols('c' + i, integer=True)

    @classmethod
    def ind_split(cls, oldind, n):
        s_oldind = str(oldind)
        return symbols('c' + str(n), integer=True)

class CodePrinter(object):
    def __init__(self, indent, indention='    '):
        self.indent = indent
        self.indention = indention
        self.buffer = ''

    def newLine(self, code):
        indented = ''
        for i in range(self.indent):
            indented += self.indention
        indented += code
        indented += '\n'
        self.buffer += indented

    def expr_str(self, s):
        return s

    def expr_int(self, i):
        return str(i)

    def expr_float(self, f):
        return str(f)

    def expr_Symbol(self, s):
        return str(s).replace('[', '(').replace(']', ')')
    expr_Mul = expr_Symbol
    expr_Add = expr_Symbol
    expr_Indexed = expr_Symbol
    expr_Zero = expr_Symbol
    expr_StrictLessThan = expr_Symbol
    expr_GreaterThan = expr_Symbol
    expr_LessThan = expr_Symbol
    expr_max = expr_Symbol
    expr_min = expr_Symbol

    def expr_Tensor(self, cmd):
        t = cmd.op
        code = t.name
        if t.is_scalar:
            return code
        code = code + '('
        code = code + ', '.join([ self.expr(a) for a in cmd.args ])
        code = code + ')'
        return code

    def expr_ASSIGN(self, cmd):
        code = self.expr(cmd.args[0]) + ' = ' + self.expr(cmd.args[1])
        return code

    def expr_CALL(self, cmd):
        fn = cmd.args[0]
        code = self.expr(fn) + "("
        code = code + ", ".join(self.expr(cmd.args[1]))
        code = code + ")"
        return code

    def expr_list(self, cmd):
        return [self.expr(x) for x in cmd]

    def expr_INSPECT(self, cmd):
        code = self.expr(cmd.args[0])
        return code

    def expr_Command(self, cmd):
        if isinstance(cmd.op, Opcode):
            fname = fname = 'expr_%s' % cmd.op.name
        else:
            fname = 'expr_%s' % type(cmd.op).__name__
        try:
            fn = getattr(self, fname)
        except AttributeError:
            raise NotImplementedError(fname)
        else:
            try:
                return fn(cmd)
            except:
                raise

    def expr(self, cmd):
        fname = 'expr_%s' % type(cmd).__name__
        try:
            fn = getattr(self, fname)
        except AttributeError:
            raise NotImplementedError(fname)
        else:
            try:
                return fn(cmd)
            except:
                raise

    def Print(self):
        print(self.buffer)

    def enterBlock(self):
        self.indent += 1

    def exitBlock(self):
        self.indent -= 1

    @contextlib.contextmanager
    def newBlock(self):
        self.indent += 1
        yield
        self.indent -= 1

"""
   (<op>, <arg>*): An op can be a first-class command operator, a function call, or a Tensor/View which can be indexed
   A Command is a Loop Free LISP-like statement.
"""
class Command(object):
    def __init__(self, op, args):
        self.op = op
        self.args = [ x for x in args ]

    def codegen(self, printer):
        if self.op == Opcode.IF:
            prologue = 'if (' + printer.expr(self.args[0]) + ') {'
            printer.newLine(prologue)
            with printer.newBlock():
                self.args[1].codegen(printer)
            if len(self.args) == 2:
                printer.newLine('}')
            else:
                raise NotImplementedError('Not support if-else yet.')
            return

        if self.op == Opcode.COND:
            n_conds = len(self.args) // 2
            for i in range(0, n_conds):
                c = 2 * i
                s = c + 1
                prologue = ('if' if i == 0 else 'else if') + ' (' + printer.expr(self.args[c]) + ') {'
                printer.newLine(prologue)
                with printer.newBlock():
                    self.args[s].codegen(printer)
                printer.newLine('}')
            return

        if self.op == Opcode.LOOP:
            loop = self.args[0]
            loop.codegen(printer)
            return 

        if self.op == Opcode.BEGIN:
            for arg in self.args:
                arg.codegen(printer)
            return

        stmt = printer.expr(self) + ';'
        printer.newLine(stmt)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if not self.op == other.op:
                return False
            if not len(self.args) == len(other.args):
                return False
            for i in range(self.args):
                if not self.args[i] == other.args[i]:
                    return False
            return True
        return False

    def replace(self, old, new):
        for pos, a in enumerate(self.args):
            if a == old:
                self.args[pos] = new
            elif isinstance(a, Command):
                a.replace(old, new)
            elif isinstance(a, (sympy.Symbol, sympy.Expr)):
                self.args[pos] = a.subs(old, new)

    def replace_if(self, target, old, new, inverse=False):
        def walk(expr):
            from functools import reduce
            if isinstance(expr, Indexed):
                cond = (expr.base == target)
                if inverse:
                     cond = not cond
                if cond:
                    return expr.subs(old, new)
                else:
                    return expr
            elif isinstance(expr, sympy.Add):
                _args = [ walk(a) for a in expr.args ]
                return reduce(operator.add, _args)
            elif isinstance(expr, sympy.Mul):
                _args = [ walk(a) for a in expr.args ]
                return reduce(operator.mul, _args)
            elif isinstance(expr, sympy.Pow):
                _args = [ walk(a) for a in expr.args ]
                return reduce(operator.pow, _args)
            elif isinstance(expr, sympy.GreaterThan):
                lhs = expr.args[0]
                rhs = expr.args[1]
                return lhs >= rhs
            elif isinstance(expr, sympy.LessThan):
                lhs = expr.args[0]
                rhs = expr.args[1]
                return lhs <= rhs
            elif isinstance(expr, int):
                return expr
            elif isinstance(expr, sympy.Symbol):
                return expr
            elif isinstance(expr, sympy.numbers.Integer):
                return expr
            else:
                debug( type(expr) )
                raise NotImplementedError(expr)

        for pos, a in enumerate(self.args):
            if a == old:
                self.args[pos] = new
            elif isinstance(a, Command):
                a.replace_if(target, old, new, inverse=inverse)
            elif isinstance(a, (sympy.Symbol, sympy.Expr)):
                self.args[pos] = walk(a)

class Opcode(Enum):
    ASSIGN = 21
    BEGIN = 31
    IF = 41
    COND = 42
    LOOP = 51
    CALL = 61
    GETITEM = 62
    SETITEM = 63
    BREAK = 71
    CONTINUE = 72
    INSPECT = 81
    pass

def Call(fn, args):
    return Command(Opcode.CALL, [ fn, args ])

def Inspect(expr):
    return Command(Opcode.INSPECT, [expr])

def Begin(args_list):
    return Command(Opcode.BEGIN, args_list)

def If(cond, then_, else_):
    if else_ is None:
        return Command(Opcode.IF, [cond, then_])
    else:
        return Command(Opcode.IF, [cond, then_, else_])

"""
   A Loop is a generalized for loop:
   for (<init>; <cond>; <step>) {
       <body>
   }
"""
class Loop(object):
    def __init__(self, init, cond, step, body):
        self.init = init
        self.cond = cond
        self.step = step
        self.body = body 

"""
   A SyntacticLoop is a perfect loop like this:
   for ( i = <start>; i < <stop>; i += <step> ) {
       for ( j = <start2>; j < <stop2>; j += <step2> ) {
           ... {
               <body>
           }
       }
   }
   A for-loop can have multiple layers, each layer iterate a dimension of a Tensor using a slice(start, stop, step).
   If <step> is less than 0, then we will replace '<' with '>' and '+=' with '-='.
   All loop restructuring operations on SyntacticLoop is based on naive syntax rewriting or pattern matching, i.e., they do not back by any polyhederal model
   , which makes it less flexible but more easy to hack.
"""
class SyntacticLoop(object):
    def __init__(self, nloops, inds, slices, body, inclusive=False):
        self.nloops = nloops
        self.inds = inds
        self.slices = slices
        self.body = body
        self.inclusive = inclusive

    @property
    def inner_loop(self):
        assert self.body.op == Opcode.LOOP
        return self.body.args[0]

    def codegen(self, printer):
        if self.inclusive:
            maybe_eq = '='
        else:
            maybe_eq = ''
        # debug('SyntacticLoop ' + str(self.nloops) + ' should == ' + str(len(self.inds)) )
        for i in range(self.nloops):
            code = 'for (int '
            code = code + str(self.inds[i]) + ' = ' + printer.expr(self.slices[i].start) + '; '
            if self.slices[i].step > 0:
                code = code + str(self.inds[i]) + ' <' + maybe_eq + ' ' + printer.expr(self.slices[i].stop) + '; '
                code = code + str(self.inds[i]) + ' += ' + printer.expr(self.slices[i].step) + ') {'
            elif self.slices[i].step < 0: 
                code = code + str(self.inds[i]) + ' >' + maybe_eq + ' ' +  printer.expr(self.slices[i].stop) + '; '
                code = code + str(self.inds[i]) + ' -= ' + printer.expr(-self.slices[i].step) + ') {'
            printer.newLine(code)
            printer.enterBlock()
        self.body.codegen(printer)
        for i in range(self.nloops):
            printer.exitBlock()
            printer.newLine('}')

    """
       Currently, blocking is *unsafe* because it does not detect boundary condition
    """
    def blocking(self, axis, step, offset=1):
        oldslice = self.slices[axis]
        oldstep = oldslice.step
        self.slices[axis] = slice(oldslice.start // step, oldslice.stop // step, 1 )
        newslice = slice(0, step, oldstep)
        oldind = self.inds[axis]
        newind = IndUtils.ind_split(oldind, self.nloops)
        self.nloops += 1
        self.inds.insert(axis+offset, newind)
        self.slices.insert(axis+offset, newslice)
        self.body.replace(oldind, oldind * step + newind)
        return self

    def reorder(self, axes):
        pass

    def unroll(self, axis, step):
        pass

    def unroll_and_jam(self, axis, step):
        pass

    def vectorize(self, axis, ramp):
        pass

    def tile(self, axes):
        pass

    def skew(self, axis, skew_axis):
        offset = self.inds[skew_axis]
        oldind = self.inds[axis]
        oldslice = self.slices[axis]
        self.slices[axis] = slice(oldslice.start + offset, oldslice.stop + offset, oldslice.step)
        self.body.replace(oldind, oldind - offset)
        return self

    def shift(self, axis, offset):
        oldind = self.inds[axis]
        oldslice = self.slices[axis]
        self.slices[axis] = slice(oldslice.start + offset, oldslice.stop + offset, oldslice.step)
        self.body.replace(oldind, oldind - offset)
        return self

    def dec_ind_start(self, axis, offset):
        assert offset > 0
        ind = self.inds[axis]
        oldslice = self.slices[axis]
        self.slices[axis] = slice(oldslice.start - offset, oldslice.stop, oldslice.step)
        self.body = Command(Opcode.IF, [ind >= oldslice.start, self.body])
        return self

    def peel(self, axis, offset):
        pass

    """
       A pack pass picks a target tensor and packs it into a new and small tensor for memory locality.
       Pack should works for affine index (though not recommended), which means it must consult the loop body to determine the size of packed tensor.
       The size of packed tensor is decided by the blocked axes, i.e., the inner loop iteration space and how loop *body* access target tensor data space.
    """
    def pack(self, blocking_axes, blocked_axes, target):
        pass

    """
       A fold assumes the loop is writing/reading a Tensor <target>. <target> is then folded into a smaller tensor, and the folded loop writing to the folded tensor instead of the default tensor.
       Example:
           for (int i = 0; i < N; i++) {
               A(i) = ...B(i)...
           }
           ---  
           for (int i = 0; i < N; i += B) {
               for (int ii = 0; ii < B; i++) {
                   A'(ii) = ...B(i+ii)...
               }
           }
           ---with halo elements = 2---
           for (int i = 0; i < N; i += B) {
               for (int ii = 0 - 2; ii < B + 2; ii++) {
                   A'(ii) = ...B(i+ii)... (B can take Out-of-Bound index)
               }
           }
           ---with halo elements = 2 and reuse---
           for (int i = 0; i < N; i += B) {
               for (int ii = (i == 0) ? -2 : 2; ii < B + 2; ii++) {
                   A'((ii + i) % (B+4)) = ...B(ii + i)... (B can take Out-of-Bound index)
               }
           }
    """
    def blocking_and_fold(self, target, axis, block, halo=0, offset=1):
        target = IndexedBase(target)
        oldslice = self.slices[axis]
        oldstep = oldslice.step
        self.slices[axis] = slice(oldslice.start // block, oldslice.stop // block, 1 )
        newslice = slice(0 - halo, block + halo, oldstep)
        oldind = self.inds[axis]
        newind = IndUtils.ind_split(oldind, self.nloops)
        self.nloops += 1
        self.inds.insert(axis+offset, newind)
        self.slices.insert(axis+offset, newslice)
        # self.body.replace(oldind, oldind * block + newind)
        self.body.replace_if(target, oldind, oldind * block + newind, inverse=True)
        self.body.replace_if(target, oldind, newind)
        return self

    def fold(self, target, axis, block, halo=0):
        pass

    """
       only safe when step == (-)1 or exists a, s.t. stop - 1 == start + a * step 
    """
    def reversal(self, axis):
        old_slice = self.slices[axis]
        assert old_slice.step == 1 or old_slice.step == -1
        self.slices[axis] = slice(old_slice.stop - 1, old_slice.start - 1, -old_slice.step)
        return self

    def fusion(self, loop2):
        assert self.nloops == loop2.nloops
        for i in range(self.nloops):
            assert self.inds[i] == loop2.inds[i]
            assert self.slices[i] == loop2.slices[i]
        self.body = Command(Opcode.BEGIN, [self.body, loop2.body])
        return self

    def fission(self, loop2):
        pass

    def parallel(self, axes):
        pass

    """
       extracts subLoops from the current loop to become a new Loop and sink the remains to body
    """
    def sink(self, sub_loops):
        assert (sub_loops < self.nloops and sub_loops > 0)
        i_start = self.nloops - sub_loops
        sub_inds = self.inds[i_start:]
        sub_slices = self.slices[i_start:]
        sub_body = self.body
        sub = SyntacticLoop(sub_loops, sub_inds, sub_slices, sub_body)
        self.nloops -= sub_loops
        debug('i_start=%d' % i_start)
        self.inds = self.inds[:i_start]
        self.slices = self.slices[:i_start]
        self.body = Command(Opcode.LOOP, [sub])
        return self

    """
       Polyhedral Extraction Tool
    """
    def pet(self):
        pass

    """
       Perform Tiling first, then fold the output tensor into a tiled tensor (with or without halo elements) so its memory can be shrinked
       The tiledand folded loop is always fused with another loop which is tiled the same way and uses its output as input, i.e., pipelined to another loop
       This is an optimization which trades working set size to parallelism.
       nextLoop is untiled by default.
    """
    @classmethod
    def tile_and_fold_and_fuse(cls, loops, axes, fold_map):
        pass


"""
   LOOP: Loop Operation & Optimization Processor
         with LISP-like syntax
"""
class LOOP:
    def __init__(self):
        pass
