import os
import sys

path = sys.argv[1]

paths = [os.path.join(path, name) for name in os.listdir(path)]

print(paths)
