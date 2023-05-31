import pandas as pd
import numpy as np
import ast


dxdy = [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6]
dz = [2e-6,5e-6,10e-6,20e-6]
maxXY = [80e-6,120e-6,160e-6,200e-6,240e-6]
maxZ = [1e-4,2e-4,3e-4,4e-4,5e-4]

str = ["dxdy","dz","maxXY","maxZ"]

df = pd.read_pickle("errors.txt")
print(df)

for s in str:
    col = df[s][0][1:-1]
    array_printout = col
    array_strings = array_printout.split("],")
    array_strings = [string.strip() + "]" for string in array_strings]

    # Convert each array string back into a NumPy array
    arrays = []
    for array_string in array_strings:
        array = np.array(ast.literal_eval(array_string))
        arrays.append(array)

    # Print the NumPy arrays
    for array in arrays:
        print(array)
