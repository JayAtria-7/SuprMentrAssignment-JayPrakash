'''
Assignment (20/02/2026)

Assignment Name : NumPy Speed Test
Description : Compare Python lists vs NumPy arrays with 1M numbers, measure execution time, write 3 observations.
'''

import time
import numpy as np
n = 1_000_000
start = time.time()
python_list = list(range(n))
python_result = [x * 2 for x in python_list]
end = time.time()
list_time = end - start
print("Python List Time:", list_time)
start = time.time()
numpy_array = np.arange(n)
numpy_result = numpy_array * 2
end = time.time()
numpy_time = end - start
print("NumPy Array Time:", numpy_time)
print("\nNumPy is faster by:", list_time - numpy_time, "seconds")