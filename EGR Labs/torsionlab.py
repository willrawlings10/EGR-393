# -*- coding: utf-8 -*-
"""TorsionLab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dj7pkDzxcRB5A11zsp0-6PLdgfNmiYhO
"""

import numpy as np
import matplotlib.pyplot as plt

#importing the torque values
torque = np.array([100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3200,3400,3600,3800,3834])
print(len(torque))
#importing the degree values
degrees = np.array([0.58,1.18,1.8,3.38,4.09,5.12,5.84,6.39,6.96,7.57,8.24,8.88,9.34,9.75,10.24,10.8,11.5,12.36,13.3,14.28,15.45,16.84,18.49,21.01,24.9,32.39,44,58.9,76,96.9,143.4,213,320,510,572])
print(len(degrees))
#import radius in m
radius = 0.009595
#import length in m
length = 0.205

"""Questions 1,2, and 3"""

#convert from degrees to radians
radians= (np.pi/180.0) * degrees
#convert torque to N*M
torque = torque*0.098
#calculating shear
strain = (radians)*(radius/length)
print(strain)

"""Question 4A: Torque vs. Strain: Full Range"""

plt.plot(strain,torque)
plt.title("Torque vs Strain: Full Range")
plt.xlabel("Strain (rad)")
plt.ylabel("Torque (N*m)")
plt.show()

"""Question 4B: Torque vs Strain: Reduced Range"""

plt.plot(strain,torque)
plt.title("Torque vs Strain: Reduced Range")
plt.xlabel("Strain (rad)")
plt.ylabel("Torque (N*m)")
plt.xlim(0,0.017)
plt.show()

"""2% Offset

Question 5:
"""

#calculating moment of inertia
inertia = (np.pi)*(0.5)*(radius**4)
rise = torque[ 1]
run = strain[1]
slope = rise/run
G = slope*(radius/inertia)
print(G)

offset_strain = strain +(strain*0.02)
plt.plot(strain, (slope+(slope*0.02))*strain, color = 'blue')
plt.plot(offset_strain, torque, color = 'green')
plt.title("Torque vs Strain: Offset")
plt.xlabel("Strain (rad)")
plt.ylabel("Torque (N*m)")
plt.xlim(0.0,0.012)
plt.ylim(0, 400)
plt.show()

#Calculating Yield Stress
yield_stress = (175.5)*(radius)/inertia
print(yield_stress)

print(torque)

"""Question 6"""

ultimate_torque = 375.732
ult_stress = (ultimate_torque *0.75*radius)/inertia
print(ult_stress)