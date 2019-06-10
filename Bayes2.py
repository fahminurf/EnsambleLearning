# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 01:48:16 2019

@author: King
"""

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score


DataTrain = pd.read_csv('TrainsetTugas4ML.csv')
DataTest = pd.read_csv('TestsetTugas4ML.csv')

#n_sample = round(len(DataTrain) * 1)

#gabung fitur dataTrain
dataTrx1=[]
dataTrx2=[]
dataTrcl=[]


for i in range (len(DataTrain)):
    dataTrcl.append(DataTrain.iloc[i]['Class'])
    dataTrx1.append(DataTrain.iloc[i]['X1'])
    dataTrx2.append(DataTrain.iloc[i]['X2'])


featuresTrain=(list(zip(dataTrx1,dataTrx2)))

#gabung fitur data Test 
dataTsx1=[]
dataTsx2=[]
dataTscl=[]


for i in range (len(DataTest)):
    dataTscl.append(DataTrain.iloc[i]['Class'])
    dataTsx1.append(DataTrain.iloc[i]['X1'])
    dataTsx2.append(DataTrain.iloc[i]['X2'])


featuresTest=(list(zip(dataTsx1,dataTsx2)))


sam1x1=[]
sam1x2=[]
sam1class=[]

sam2x1=[]
sam2x2=[]
sam2class=[]

sam3x1=[]
sam3x2=[]
sam3class=[]

#buat sample data untuk model

#sample1
for i in range (298):
    rand=np.random.randint(0,297)
    sam1class.append(DataTrain.iloc[rand]['Class'])
    sam1x1.append(DataTrain.iloc[rand]['X1'])
    sam1x2.append(DataTrain.iloc[rand]['X2'])

#sample2
for i in range (298):
    rand=np.random.randint(0,297)
    sam2class.append(DataTrain.iloc[rand]['Class'])
    sam2x1.append(DataTrain.iloc[rand]['X1'])
    sam2x2.append(DataTrain.iloc[rand]['X2'])
    
#sample3
for i in range (298):
    rand=np.random.randint(0,297)
    sam3class.append(DataTrain.iloc[rand]['Class'])
    sam3x1.append(DataTrain.iloc[rand]['X1'])
    sam3x2.append(DataTrain.iloc[rand]['X2'])




#buat model
GausNB= GaussianNB()


#    gabung x1 dengan x2

features1=(list(zip(sam1x1,sam1x2)))
features2=(list(zip(sam2x1,sam2x2)))
features3=(list(zip(sam3x1,sam3x2)))

##buat model
model1 = GausNB.fit(features1,sam1class)

model2 = GausNB.fit(features2,sam2class)

model3 = GausNB.fit(features3,sam3class)

#cek akurasi
cekakurmod1 = model1.predict(features1)
cekakurmod2 = model2.predict(features2)
cekakurmod3 = model3.predict(features3)

    
print('AKURASI DARI MODEL 1 :',accuracy_score(cekakurmod1,sam1class)*100,'%')
print('AKURASI DARI MODEL 2 :',accuracy_score(cekakurmod2,sam2class)*100,'%')
print('AKURASI DARI MODEL 3 :',accuracy_score(cekakurmod2,sam3class)*100,'%')

#predict to models

ymodel1 = model1.predict(featuresTrain)
ymodel2 = model2.predict(featuresTrain)
ymodel3 = model3.predict(featuresTrain)



#summing y models

mod1 = pd.DataFrame(ymodel1)
mod1df= mod1.rename(index=str, columns={0:'ymod1'})

mod2 = pd.DataFrame(ymodel2)
mod2df= mod2.rename(index=str, columns={0:'ymod2'})

mod3 = pd.DataFrame(ymodel3)
mod3df= mod3.rename(index=str, columns={0:'ymod3'})

merged = pd.concat([mod1df,mod2df,mod3df],axis =1)

#making sum and sign
ahay = merged.mode(axis=1)




sumY=[]
signY=[]
for i in range(len(merged)):
    sumY.append(int(merged.iloc[i][0]+merged.iloc[i][1]+merged.iloc[i][2]))

sumYdf= pd.DataFrame(sumY)

for i in range(len(sumY)):
    if (sumYdf.iloc[i][0]== 3):
        signY.append(1)
    else:
        signY.append(2)

print('AKURASI DARI MODEL GABUNGAN TERHADAP DATA TRAIN: ' ,accuracy_score(DataTrain.iloc[:,2],signY))

#testing models  to data test

yfinalmodel1 = model1.predict(featuresTest)
yfinalmodel2 = model2.predict(featuresTest)
yfinalmodel3 = model3.predict(featuresTest)

yfinalmodel1df = pd.DataFrame(yfinalmodel1)
yfinalmodel2df = pd.DataFrame(yfinalmodel2)
yfinalmodel3df = pd.DataFrame(yfinalmodel3)



mergedfinal = pd.concat([yfinalmodel1df,yfinalmodel2df,yfinalmodel3df],axis =1)

modusfinal = mergedfinal.mode(axis=1)

sumYfinal=[]
signYfinal=[]

#summing y test
for i in range (len(yfinalmodel1)):
    sumYfinal.append(int(yfinalmodel1[i]+yfinalmodel2[i]+yfinalmodel3[i]))

#signing y test 

for i in range (len(yfinalmodel1)):
    if (sumYfinal[i] == 3):
        signYfinal.append(1)
    else:
        signYfinal.append(2)


FinallY1 = pd.DataFrame(signYfinal)
FinallY = FinallY1.rename(index=str, columns={0:'Class'})


FinallY.to_csv('TebakanTugas4ML.csv',index=None,header=None)







