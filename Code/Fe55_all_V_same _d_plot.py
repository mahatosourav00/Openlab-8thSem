from operator import le
from certifi import where
import numpy as np
import matplotlib.pyplot as plt
from my_library import *
from scipy.optimize import curve_fit




def func_gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp(-((x - ctr)**2)/(2*(wid)**2))
    return y

voltages = [2375,2425,2450,2475, 2500,2525,2550,2575,2600]
#file = open('Voltage_FWHM_10cm.txt','w')






max_x = []
max_y = []

for row in range(len(voltages)):
    data = open('../Data/April-25_Data/Fe55_d10cm_%sV_600s.Spe' %voltages[row],'r')
    data_counts = spe_file_read(data)
    channels = np.arange(0,16384,1)

    data1 = open("../Data/April-25_Data/Background_d10cm_%sV_600s.Spe" %voltages[row],'r')
    back_counts = spe_file_read(data1)
    #print(back_counts)
    data_counts = np.array(data_counts)
    back_counts = np.array(back_counts)
    #print(back_counts)

    corr_data = data_counts - back_counts
    for i in range(len(corr_data)):
        if corr_data[i] < 0:
            corr_data[i] = 0
    channels = channels[200:]
    corr_data = corr_data[200:]
    #plt.plot(channels,corr_data)
    plt.plot(channels,corr_data)
    #sort = np.sort(corr_data)
    #maxEle = sort[-1]
    #secmaxEle = sort [-2]
    #maxIndex = np.where(corr_data == maxEle)
    #secmaxIndex = np.where(corr_data == secmaxEle)

    guess = [[870, 2000, 20, 1790, 13580, 35],
            [1320, 1250, 20, 2850, 9580, 35],  #2400

            [1500, 1150, 20, 3220, 8110, 35],   #2450
            [1850, 950, 20, 4000, 6660, 355],   #2475
            [2560, 750, 20, 5000, 4980, 15],   #2500
            [3200, 570, 20, 6370, 3900, 55],    #2525
            [3380, 500, 20, 7000, 3550, 55],    #2550
            [3980, 388, 50, 8400, 2800, 55],    #2575
            [4550, 320, 20, 9650, 2410, 200]]  #2600

    popt, pcov = curve_fit(func_gauss, channels, corr_data, p0=guess[row])
    #print(popt)
    fit = func_gauss(channels, *popt)
    popt1 = popt[:3]
    popt2 = popt[3:]

    fit1 = func_gauss(channels, *popt1)
    fit2 = func_gauss(channels, *popt2)

    FWHM2 = 2*np.sqrt(2*np.log(2))*popt[5]
    max_x.append(popt2[0])
    max_y.append(popt2[1])
    
    #file.write(str(voltages[row])+'\t'+str(FWHM2)+'\n')

    print('FWHM=',FWHM2)

    #plt.plot(channels, fit1)
    #plt.plot(channels, fit2)
    plt.plot(channels, fit,label = '%sV' %voltages[row])
    #plt.axvline(x = popt[0],color = 'b',ls='--')
    #plt.text(popt[0]+80,popt[1]+500 ,'2nd maxima at %s' %int(popt[0]))
    #plt.axvline(x = popt[3],color = 'r',ls ='--')
    #plt.text(popt[3]+80,popt[4]-1000 ,'1st maxima at %s' %int(popt[3]))
    plt.legend()
    plt.xlabel("Channel no.")
    plt.ylabel("Counts")
    plt.title('Plot for Fe55 kept at different Voltages and 10cm')
    #plt.xlim(350,popt[3]+5000)



#
#def funcinv(x,a):
#    return a/(x)
#
#popt3,pocv = curve_fit(funcinv, max_x, max_y)



#plt.plot(max_x,max_y)
#popt3 = curve_fit(poly,max_x,max_y)
#print(popt3)
#tfit = funcinv(channels,*popt3)
#plt.plot(channels,tfit,'--',label='Inverse fit')
plt.savefig('Plot_Fe55_All_V_same_d_600s.png')
#plt.ylim(0,14000)
#plt.legend()
plt.show()
plt.close()



#file.close()