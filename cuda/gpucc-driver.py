import sys
from subprocess import call

verbose = True

with open("cmd.ins", "r") as instructions:
  for inst in instructions:
    if inst.isspace():
      continue

    inst = inst.lstrip().replace("$FILENAME", "gaussian_elimination").replace('"', '')
    if verbose:
      print(inst)
    if inst[0] == '#':
      continue

    tokens = inst.split()
    # print(tokens)
    if call(tokens) != 0:
      print("FATAL: driver exit.")
      exit()

print("driver finish successfully!")    
