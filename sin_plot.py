import matplotlib.pyplot as plt

import numpy as np 


x = np.linspace(2,20,50)

y = x.flat 

emp= []

for xs in y :

	emp.append(xs)


sin1 = np.sin(emp)


plt.plot(emp,sin1)

plt.title("sin plot")

plt.show()

