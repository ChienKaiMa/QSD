import csv
import numpy as np

# https://www.geeksforgeeks.org/convert-numpy-array-into-csv-file/

import pandas as pd 
import numpy as np 
  
# create a dummy array 
arr = np.arange(1,11).reshape(2,5) 
print(arr) 
  
# convert array into dataframe 
df = pd.DataFrame(arr) 
  
# save the dataframe as a csv file 
df.to_csv("data1.csv")


import numpy as np 
arr = np.arange(1,11) 
print(arr) 

# use the tofile() method 
# and use ',' as a separator 
arr.tofile('data2.csv', sep = ',')


## # Save an array
## fname = f"newfile.csv"
## np.savetxt(fname, [3, 4, 5], fmt="%.16f", delimiter=",")
