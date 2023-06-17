
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from datetime import datetime, timedelta

def formatrupiah(uang):
    y = str(uang)
    if len(y) <= 3 :
        return 'Rp ' + y     
    else :
        p = y[-3:]
        q = y[:-3]
        return   formatrupiah(q) + '.' + p
        print ('Rp ' +  formatrupiah(q) + '.' + p )


def bdesmodel(data, alpha, slope, konstanta, pred_test,panjangprediksi):
  s1 = np.zeros(len(data))                       #variabel untuk menyimpan nilai pemulusan pertama
  s2= np.zeros(len(data))                       #variabel untuk menyimpan nilai pemulusan kedua
  prediksi = np.zeros(len(data))
  prediksi[0] = data['Harga'][1]
  for i in range(len(data)):
    if i < len(data)-1:
      if i < 1:
        s1[i] = data['Harga'][i]
        s2[i]= data['Harga'][i]
        konstanta[i] = 2*s1[i] - s2[i] 

      else:
        s1[i] = (alpha*data['Harga'][i]) +( (1-alpha) * s1[i-1])
        s2[i] = (alpha*s1[i] )+ ((1-alpha) * s2[i-1])
        konstanta[i] = (2*s1[i]) - s2[i]

      slope[i] = (alpha/(1-alpha)) * (s1[i] - s2[i])
      prediksi[i+1]= round(konstanta[i] + slope[i]*1)
      
    if i == len(data)-1:
        s1[i] = alpha*data['Harga'][ i] + (1-alpha) * s1[i-1]
        s2[i] = alpha*s1[i] + (1-alpha) * s2[i-1]
        konstanta[i] = 2*s1[i] - s2[i]
        slope[i] = alpha/(1-alpha) * (s1[i] - s2[i])


  for i in range(panjangprediksi-1):
    if i == 0:
      pred_test[i]=round( konstanta[len(slope)-1]+ slope[len(slope)-1] *(i+1))
      pred_test[i+1]= round(konstanta[len(slope)-1]+ slope[len(slope)-1] *(i+2))
    else :
      pred_test[i+1]= round(konstanta[len(slope)-1]+ slope[len(slope)-1] *(i+2))

  return prediksi

def pred_periodedepan(kons,slope,panjangprediksi, datates):
   prediksi = []
   temp =len( datates) -1
   for i in range(panjangprediksi):
      if i == 0:
        prediksi.append(round( kons[len(slope)-1]+ slope[len(slope)-1] *temp))
        temp = temp +1
      else :
        prediksi.append(round(kons[len(slope)-1]+ slope[len(slope)-1] *temp))
        temp = temp +1
   return prediksi

def tanggal_kedepan(testdata, periodedepan):
  temp = testdata["Tanggal"][845]
  for i in range(len(periodedepan)+1):
    if i == 0:
      periodedepan["Tanggal"][i] = temp + timedelta(days=1)
    else:
      periodedepan["Tanggal"][i] =  periodedepan["Tanggal"][i-1]  + timedelta(days=1)

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def golden_section(data,test, test_minyak,a, d, slope, konstanta, panjangprediksi,tol=0.0000001):
  iter = 50
  r = (-1 +math.sqrt(5)) /2
  k = 0
  b = r * a + (1 - r) *d
  c = a+d - b

  while ((abs(d-a) > tol) and (k < iter)):
    k  = k+1

    fb = bdesmodel(data, b, slope, konstanta, test_minyak,panjangprediksi)
    mapet_fb = mape(test['Harga'], test_minyak)
    fc = bdesmodel(data,c, slope, konstanta,test_minyak,panjangprediksi)
    mapet_fc = mape(test['Harga'], test_minyak)

    if (  (mapet_fb < mapet_fc)):
      d = c
      c = b
      b = r * a + (1 - r) * d
    else:
      a = b 
      b = c
      c = a+ d - b
  return  (d + a) / 2

  
