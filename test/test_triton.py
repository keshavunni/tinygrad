import torch
import triton
import triton.language as tl
from triton.compiler import compile, instance_descriptor
from triton.runtime import JITFunction
import unittest

# check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class TestOps(unittest.TestCase):
  def _program(self, b0, b1, b2):
    idx = tl.program_id(0)
    x = tl.load(b1 + idx)
    y = tl.load(b2 + idx)
    tl.store(b0 + idx, x+y)
  
  def test_program(self):
    program_jit = JITFunction(self._program)
    # JITFunction(self._program) {'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32'}, 'device': 0, 'constants': {}, 'num_warps': 4, 'num_stages': 3, 'extern_libs': None, 'configs': (instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),)}
    # ast -> ttir -> ttgir -> llir -> ptx -> cubin
    compiled = compile(program_jit, signature={0: '*fp32', 1: '*fp32', 2: '*fp32'}, cc=30, device="cuda:0", constants={}, num_warps=4, num_stages=3, extern_libs=None, configs=(instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),))
    # compiled = compile(program_jit, signature={0: '*fp32', 1: '*fp32', 2: '*fp32'}, device="cuda:0")
    print(compiled.asm['ast'])
    print(compiled.asm['ttir'])
    print(eval(compiled.asm['llir']).decode('utf-8'))
