import cgraph

cp = cgraph.CodePrinter(1)
m_in = cgraph.MatrixXd(2048, 3072, 'in')

stage1 = cgraph.Save( cgraph.Conv1D(m_in, 1, filter_weights={-1:1, 0:1, 1:1}), 'blurx' )
stage2 = cgraph.Save( cgraph.Conv1D(stage1, 0, filter_weights={-1:1, 0:1, 1:1}), 'out' )

l1 = stage1.lower().blocking_and_fold('blurx', 0, 32, halo=1, offset=2).blocking_and_fold('blurx', 1, 32, offset=2).shift(2, -1).sink(1)
l2 = stage2.lower().blocking_and_fold('blurx', 0, 32, offset=2).blocking_and_fold('blurx', 1, 32, offset=2).sink(1).dec_ind_start(2, 2)

l1.fusion(l2).codegen(cp)

# cp.Print()

import numpy as np

c_in = cgraph.Tensor( (2,3,4), np.dtype('float64'), name='c_in' )
d_in = cgraph.Tensor( (2,3,4), np.dtype('float64'), name='d_in' )
c_in_1 = cgraph.Tensor( (3,1), np.dtype('float64'), name='c_in_1' )
c_in_2 = cgraph.Tensor( (2,3), np.dtype('float64'), name='c_in_2' )

test_collapse = cgraph.Save( cgraph.Collapse(c_in, [0, 1]), 'save')
test_collapse.lower().codegen(cp)

test_collapse = cgraph.Save( cgraph.Collapse(c_in, [0, 1, 2]), 'save')
test_collapse.lower().codegen(cp)

test_collapse = cgraph.Save( cgraph.Collapse(c_in, [1, 2]), 'save')
test_collapse.lower().codegen(cp)

test_collapse = cgraph.Collapse(c_in, [0])

test_concat = cgraph.Concat( [c_in, cgraph.Broadcast(c_in_1, (2,3,2))], 2, 'Concat' )
test_concat.lower().codegen(cp)

test_dot = cgraph.Dot(c_in_2, c_in_1, "Dot")
test_dot.lower().codegen(cp)

test_add = cgraph.Save( c_in + d_in + c_in + d_in, name='add' )
test_add.lower().codegen(cp)

from cloog import *
from sympy import floor

s = Statement([I, J], "0 <= i < 100 and 0 <= j < 100", S(1, [I, J]), schedule=[I])
ss =  Statement([I, J, K], "0 <= i < 100 and 0 <= j < 100 and 0 <= k < 50", S(2, [I, J, K]), schedule=[I+J])

ms = MultiStatements([s, ss])
ms.isl_codegen(cp)

cp.Print()
ms.dump_isl_code()
