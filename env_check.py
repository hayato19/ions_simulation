import sys

print("Python executable:")
print(sys.executable)

print("\nPython path:")
for p in sys.path:
    print(p)

print("\nImport check:")
try:
    import numpy
    print("numpy:", numpy.__version__)
except Exception as e:
    print("numpy error:", repr(e))

try:
    import qutip
    print("qutip:", qutip.__version__)
except Exception as e:
    print("qutip error:", repr(e))