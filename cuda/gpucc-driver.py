#! /usr/bin/python3

import sys
from subprocess import call

import argparse
parser = argparse.ArgumentParser(description='Invoke gpucc via clang explicitly.')
parser.add_argument('--script', '-s', metavar='<script>', nargs='?', help='Scripts for compile instrunctions.', default='cmd.ins')
parser.add_argument('--filename', '-f', metavar='<file>.cu', nargs='?', help='CUDA filename.', default='gaussian_elimination')
parser.add_argument('--output', '-o', metavar='<output>', nargs='?', help='Ouput filename.', default='a.out')
parser.add_argument('--verbose', '-v', action='count')

args = parser.parse_args()

Verbose = False
if args.verbose:
  Verbose = args.verbose > 0

with open(args.script, "r") as instructions:
  for inst in instructions:
    if inst.isspace():
      continue

    inst = inst.lstrip().replace('"', '')
    inst = inst.replace("$FILENAME", args.filename)
    inst = inst.replace("$OUTPUT", args.output)

    if Verbose:
      print(inst)
    if inst[0] == '#':
      continue

    tokens = inst.split()
    # print(tokens)
    if call(tokens) != 0:
      print("FATAL: gpucc-driver exit when " + inst)
      exit()

print("gpucc-driver finish successfully!")    
