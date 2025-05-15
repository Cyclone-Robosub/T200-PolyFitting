Python Function
A simple polyfit from Python was created from the publicly available T-200 data (attached in this repository), including fitting to a zero band between 1460 and 1560.
The coefficients for the PWM values above and below this zero band are then printed out (with values in between set to zero), as well as an MSE value for error. Plots of 
The error, polyfit at each voltage, and the plot of the actual values from the dataset are also made for visual representation of the plot.

MATLAB Function
The coefficients are taken from the Python function and put into a function centered at 1500. This MATLAB function can then take a value for voltage and PWM to output a
corresponding force value, with piecewise components based on the PWM (created as an upper and lower index for the function, including an index between -40 and 40 for the
zeroband). The base function is below, and the 3D plot is found below.



Function for Voltage V and PWM Centered at 1500 P:
```
P < -40
force = 0.18450859598294592 - 0.03877245224750089 * V + 0.00437326839182978 * V^2 - 0.00014159192815009454 * V^3 + 0.0047430745735151875 * P - 0.0011779727652119001 * P * V
+ 0.00012626694389350853 * P * V^2 - 3.6521719807495034e-06 * P * V^3 - 3.394361152608232e-05 * P^2 + 7.081135717273334e-06 * P^2 * V - 6.937006893095161e-07 * P^2 * V^2
+ 1.7574133886452198e-08 * P^2 * V^3 - 1.0125491849106606e-07 * P^3 + 2.3956832126051456e-08 * P^3 * V - 2.0262771883546578e-09 * P^3 * V^2 + 5.312516822050221e-11 * P^3 * V^3

P > 40
force = - 2.2103591356566494 + 0.5020761226661944 * V - 0.038894245021323104 * V^2 + 0.0009685515042413979 * V^3 + 0.04287528029632993 * P - 0.010018667907150527 * P * +
0.0007918978226055346 * P * V^2 - 1.965296626711245e-05 * P * V^3 - 0.00015245579360787697 * P^2 + 3.5803548508328624e-05 * P^2 * V - 2.3976087919487086e-06 * P^2 * V^2 +
5.395703879569961e-08 * P^2 * V^3 + 1.3322303527138988e-07 * P^3 - 2.9258256623955067e-08 * P^3 * V + 1.7469073167777621e-09 * P^3 * V^2 - 3.0349724016130765e-11 * P^3 * V^3

-40 < P < 40
force = 0
```
