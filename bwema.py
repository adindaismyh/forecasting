import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import math
from bdes import mape

def bwema(data, alpha, slope, konstanta, pred_test,nilai_Bt):
  s1 = np.zeros(len(data))                       #variabel untuk menyimpan nilai pemulusan pertama
  s2= np.zeros(len(data))                       #variabel untuk menyimpan nilai pemulusan kedua
  prediksi = np.zeros(len(data))
  prediksi[0] = data.iloc[0][1]
  for i in range(len(data)):
    if i <len(data)-1:
      if i < 22:
        s1[i] = alpha*data.iloc[i][ 1] + (1-alpha) * data.iloc[i][ 1]
        s2[i] = alpha*s1[i] + (1-alpha) * data.iloc[i][ 1]
        konstanta[i] = 2*s1[i] - s2[i]
        
      else :
        s1[i] = alpha*data.iloc[i][ 1] + (1-alpha) * nilai_Bt["Bt"][i]
        s2[i] = alpha*s1[i] + (1-alpha) * nilai_Bt["Bt"][i]
        konstanta[i] = 2*s1[i] - s2[i]

      slope[i] = alpha/(1-alpha) * (s1[i] - s2[i])
      prediksi[i+1]= round(konstanta[i] + slope[i]*1)

    if i == len(data)-1:
      s1[i] = alpha*data.iloc[i][ 1] + (1-alpha) * nilai_Bt["Bt"][i]
      s2[i] = alpha*s1[i] + (1-alpha) * nilai_Bt["Bt"][i]
      konstanta[i] = 2*s1[i] - s2[i]
      slope[i] = alpha/(1-alpha) * (s1[i] - s2[i])
  
  for i in range(len(pred_test)-1):
    if i == 0:
      pred_test[i]=round( konstanta[len(slope)-1]+ slope[len(slope)-1] *(i+1))
      pred_test[i+1]= round(konstanta[len(slope)-1]+ slope[len(slope)-1] *(i+2))
    else :
      pred_test[i+1]= round(konstanta[len(slope)-1]+ slope[len(slope)-1] *(i+2))
    
  
  return  prediksi

def golden_section2(data,test, test_minyak,a, d, slope, konstanta,  nilai_Bt, tol=0.0000001):
  iter = 50
  r = (-1 +math.sqrt(5)) /2
  k = 0
  b = r * a + (1 - r) *d
  c = a+d - b

  while ((abs(d-a) > tol) and (k < iter)):
    k  = k+1
    fb = bwema(data, b, slope, konstanta, test_minyak, nilai_Bt)
    mape1 = mape(data['Harga'],fb) 
    mapet1 = mape(test['Harga'], test_minyak)
    fc = bwema(data,c, slope, konstanta,test_minyak, nilai_Bt)
    mape2 = mape(data['Harga'],fc) 
    mapet2 = mape(test['Harga'], test_minyak)

    if ((mapet1 < mapet2)):
      d = c
      c = b
      b = r * a + (1 - r) * d
    else:
      a = b
      b = c
      c = a+ d - b
  return  (d +a) / 2

def nilaiBt (weights, data, Bt) :
  n = len(weights)
  temp = 0
  for  i  in range (len(data)):
    if i == n:
      for j in range(len(weights)):
        temp = temp +(weights[j] * data["Harga"][n - j - 1])
      Bt.append(temp/sum(weights))
      n = n + 1 
    else:
      Bt.append(0)
    temp = 0
