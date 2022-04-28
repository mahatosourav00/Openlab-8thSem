from certifi import where
import numpy as np
import matplotlib.pyplot as plt
from my_library import *
from scipy.optimize import curve_fit

data = open('../Data/April-25_Data/Fe55_d10cm_2525V_600s.Spe','r')
data_counts = spe_file_read(data)
channels = np.arange(0,16384,1)

data1 = open("../Data/April-25_Data/Background_d10cm_2525V_600s.Spe",'r')
back_counts = spe_file_read(data1)
#print(back_counts)
data_counts = np.array(data_counts)
back_counts = np.array(back_counts)
#print(back_counts)

corr_data = data_counts - back_counts
#print(corr_data)
for i in range(len(corr_data)):
    if corr_data[i] < 0:
        corr_data[i] = 0
channels = channels[200:]
corr_data = corr_data[200:]
plt.plot(channels,corr_data)
#plt.plot(channels,back_counts)



def func_gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp(-((x - ctr)**2)/(2*(wid)**2))
    return y

sort = np.sort(corr_data)
maxEle = sort[-1]
secmaxEle = sort [-2]
maxIndex = np.where(corr_data == maxEle)
secmaxIndex = np.where(corr_data == secmaxEle)

guess = [3200, 570, 20, 6370, 3900, 55]

popt, pcov = curve_fit(func_gauss, channels, corr_data, p0=guess)
print(popt)
fit = func_gauss(channels, *popt)
popt1 = popt[:3]
popt2 = popt[3:]

fit1 = func_gauss(channels, *popt1)
fit2 = func_gauss(channels, *popt2)
print(func_gauss(4000,*popt2))

FWHM2 = 2*np.sqrt(2*np.log(2))*popt[5]
print('FWHM=',FWHM2)
plt.plot(channels, fit1)
plt.plot(channels, fit2)
plt.plot(channels, fit)
plt.show()