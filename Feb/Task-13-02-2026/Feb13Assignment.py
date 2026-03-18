'''
create numpy array with the following temperature(celcius)[28,32,30,37,36,38]
Requirements:

convert the list into numpy array
print maximum and minimum temp
calculate average temp
display last 3 days temp

'''

import numpy as np
arr=np.array([28,32,30,37,36,38])
print("Maximum is :",np.max(arr))
print("Minimum is :",np.min(arr))
print("Average is :",np.mean(arr))
print(arr[-3:])
