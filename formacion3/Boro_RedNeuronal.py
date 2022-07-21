#!/usr/bin/env python
# coding: utf-8

# ## RAF - RED NEURONAL PARA BORO - Rev 1 - 05-11-2020
# ## DENSAMENTE INTERCONECTADA
# ### Herramienta de Generación

# In[1]:


#Library import
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import math 
import sys
print(tf.__version__)

from clearml import Task

task = Task.init(project_name="Acciona-Formacion_boro", task_name="tarea 1")

from clearml import Dataset

dataset = Dataset.create(
    dataset_name="boro_pilar",
    dataset_project="formacion2"
)
dataset.add_files(
    path="boro.csv",
)
dataset.upload()
dataset.finalize()


# In[2]:


#ALL BELOW FUNCTIONS ONLY USED TO PLOT RESULT

def movingaverage(vect,window):
    #moving average of a data serie.
    #window is the number of element to compute de average
    
    result=np.zeros(len(vect)-window+1)
    for ii in range(0,window):
        result=result+vect[0+ii:len(vect)-window+1+ii]
    result=result/window
    return result



def count(list1, r): 
    #obtain the number of elements wich are lesser than r
    
    c = 0 
    
    M=np.shape(list1)
    M=M[0]
    
    # traverse in the list1 
    for x in list1: 
        # condition check 
        if x<= r: 
            c+= 1 
    per=100*c/M
    return c,per



def print_result(yy_pred,yvs,savenum):
    #Given the predicted data and the real data plot and save:
    
    #correlation between the variables
    pearsoncoeff = np.corrcoef(  np.transpose(np.concatenate((yy_pred,yvs), axis=1))   )
    plt.scatter(yvs,yy_pred)#X,Y
    plt.title('y predicted vs y meaured. Pearson Coeff: '+str(pearsoncoeff[0,1]))
    plt.xlabel("y measured")
    plt.ylabel("y predicted")
    plt.savefig(str(savenum)+'pearsoncoef.png')
    plt.show()
   

    #absolute error distribuction
    #____________________________________________________________________
    err=yy_pred-yvs
    
    texto="Distribución del error abs \n media de la distribucion del error absoluto: "+str( np.mean(err))+"\n error medio: "+str(np.mean(abs(err)))+"\n desviacion estandar del error: "+ str(np.std(err))
    plt.hist(err, bins='auto')  
    plt.title(texto )
    plt.savefig(str(savenum)+'err.png')
    plt.show()

    
    #relative error distribution
    #____________________________________________________________________
    ind=np.where(yvs!=0)[0]
    eer_rel=err[ind]/yvs[ind]
    
    texto2="Distribución del error relativo \n media de la distribucion del error relativo medio: "+str( np.mean(eer_rel))+"\n error relativo medio: "+str(np.mean(abs(eer_rel)))+"\n desviacion estandar del error relativo: "+ str(np.std(eer_rel))
    plt.hist(eer_rel, bins='auto')  
    plt.title(texto2)
    plt.savefig(str(savenum)+'err_rel.png')
    plt.show()

    
    #data percentage well stimated depending on the relative error allowed
    #____________________________________________________________________
    dim=50
    per_gui=np.zeros((dim))
    deg=np.zeros((dim))
    cont=0
    for ii in range(1,dim+1):
        deg[cont]=2*ii
        r=2*ii
        d11,d12=count(eer_rel*100, r)
        per_gui[cont]=d12
        cont=cont+1


    plt.plot(deg, per_gui, 'bo',label="% imagenes") 
    plt.title('Porcentaje de datos estimados correctamente en funcion del porcentaje de error relativo permitido')
    plt.xlabel('Porcentaje de error relativo permitido')
    plt.ylabel('Porcentaje de datos estimados correctamente')
    plt.legend()
    plt.savefig(str(savenum)+'porcentajes_acierto.png')
    plt.show()
    
    return 


def clasific(yy_pred,yvs,marg):
    bien_est=0
    infraestimadas=0
    sobreestimadas=0
    for ii in range(0,yy_pred.shape[0]):
        if yy_pred[ii]>(1+marg)*yvs[ii]:
            sobreestimadas=sobreestimadas+1
        elif yy_pred[ii]<(1-marg)*yvs[ii]:
            infraestimadas=infraestimadas+1
        else:
            bien_est=bien_est+1
    print("bien estimadas: ", bien_est)
    print("sobre estimadas: ",sobreestimadas)
    print("infraestimadas: ", infraestimadas)
    
    return

def print_hist(historial,savenum):

    history_dict = historial.history
    history_dict.keys()
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)

    #training loss
    #_______________________________________________________________________________________________
    plt.plot(epochs, loss, 'bo', label='Training loss') # "bo" is for "blue dot"
    plt.title('training loss'),plt.xlabel('Epochs'),plt.ylabel('Loss')
    plt.legend()
    plt.savefig(str(savenum)+'_trainingloss.png')
    plt.show()

    
    #validation loss
    #_______________________________________________________________________________________________
    plt.plot(epochs, val_loss, 'bo', label='Training acc') 
    plt.title('validation loss'),plt.xlabel('Epochs'),plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(str(savenum)+'_validationloss.png')
    plt.show()

    #training and validation loss
    #_______________________________________________________________________________________________
    plt.plot(epochs[2:-1], loss[2:-1], 'b.',linewidth=1, label='Training loss') # "bo" is for "blue dot"
    plt.plot(epochs[2:-1], val_loss[2:-1], 'g.',linewidth=1, label='Validation loss acc')
    #plt.plot(epochs, loss, 'bo', label='Training loss') # "bo" is for "blue dot"
    #plt.plot(epochs, val_loss, 'go', label='Validation loss acc') 

    plt.title('validation loss'),plt.xlabel('Epochs'),plt.ylabel('Accuracy')
    plt.legend()


    plt.savefig(str(savenum)+'_validation_training_loss.png')
    plt.show()
    
    
    #training and validation loss with mooving average
    #_________________________________________________________________________________
    
    Wind=25
    
    plt.plot(epochs[2:-Wind], movingaverage(loss[2:-1],Wind), 'b.',linewidth=1, label='Training loss') # "bo" is for "blue dot"
    plt.plot(epochs[2:-Wind], movingaverage(val_loss[2:-1],Wind), 'g.',linewidth=1, label='Validation loss acc')
    #plt.plot(epochs, loss, 'bo', label='Training loss') # "bo" is for "blue dot"
    #plt.plot(epochs, val_loss, 'go', label='Validation loss acc') 

    plt.title('Val loss acc W Mooving AVG'),plt.xlabel('Epochs'),plt.ylabel('Accuracy')
    plt.legend()


    plt.savefig(str(savenum)+'_validation_training_loss_average.png')
    plt.show()
 
    return


# ### CONFIGURACION DEL SET DE DATOS Y LAPSO TEMPORAL

# In[11]:


# ---------------------------
# SET DE DATOS PARA EL MODELO
# ---------------------------
#relative Path to the folder
# fullpath = "C:/Users/mamartint/OneDrive - ACCIONA S.A/MachineLearnig/DataSets/RAF/Boro/RO7 VIRTUAL BORON RAF-OCT 2020_MMT_SinOut.csv"   
fullpath = "boro.csv"
# ------------------------------------------
# ¿SE REALIZA LA NORMALIZACIÓN DE LOS DATOS?
# ------------------------------------------
normalizacion=1

# ------------------------------------------
# PASOS TEMPORALES HACIA ATRAS
# ------------------------------------------
#HOW many backsteps can be considered
backsteps=1

# ------------------------------------------
# SELECCION DE VARIABLES PARA EL MODELO
# ------------------------------------------
selector_variable=np.array([9,11,12,15])

# ------------------------------------------
# TAMAÑO DEL SET DE ENTRENAMIENTO
# ------------------------------------------
train_percentage=0.8


# In[12]:


#LOADING THE DATA
data = pd.read_csv(fullpath)
lista = data.columns.values.tolist()
print (lista[10], " / ", lista[12], " / ",lista[13], " / ",lista[16])


# In[14]:


#LOADING THE DATA
data = pd.read_csv(fullpath)


#se elimina la primera columna, la de la Fecha, ya que no es relevante
data = data.drop(data.columns[[0]], axis='columns')
data.head()

Xc = data.columns.tolist()


#from pandas data to numpy array
npdata=data.to_numpy()


#the data normalizacion is selected

if normalizacion==1:
    maximus=np.amax(npdata,axis=0)
    #print(maximus)
    minimun=np.amin(npdata,axis=0)
    #print(minimun)

    print(" ")
    npdata=(npdata-minimun)/(maximus-minimun)
    print(npdata[0,:])
    
elif normalizacion==2:
    maximus=np.amax(npdata,axis=0)
    datamean=np.mean(npdata,0)
    print(datamean)
    npdata=(npdata-datamean)/maximus
    print(npdata[0,:])
    
else:
    pass





fulldata=npdata
for ii in range(1,backsteps+1):
    fulldata=np.concatenate((fulldata[1:,:], npdata[0:-ii,:]), axis=1)
    
print(fulldata.shape)
print(" ")




#THE DATA MATRIX is row randomized
np.random.shuffle(fulldata)


sep=round(train_percentage*fulldata.shape[0])
N_entr=selector_variable.shape

xx_training_set=fulldata[0:sep,selector_variable]
yy_training_set=fulldata[0:sep,0:1]
xx_validation_set=fulldata[sep:,selector_variable]
yy_validation_set=fulldata[sep:,0:1]

print(np.shape(xx_training_set))
print(np.shape(yy_training_set))
print(np.shape(xx_validation_set))
print(np.shape(yy_validation_set))


# In[15]:


print (maximus)


# In[16]:


print(minimun)


# In[17]:


print ("Boro: ", maximus[0], "   COND: ", maximus[9], "   PRES: ", maximus[11], "   PH: ", maximus[12], "   FLOW: ", maximus[15])


# In[18]:


print ("Boro: ", minimun[0], "   COND: ", minimun[9], "   PRES: ", minimun[11], "   PH: ", minimun[12], "   FLOW: ", minimun[15])


# In[19]:


#Only run the fist time to generate and storage the variable
#it is used to name the storage data
savenum=60
np.save("savenum",savenum)


# ### DEFINCICIÓN DE LA RED NEURONAL

# In[20]:


savenum=np.load("savenum.npy")

# Create a basic model instance
#model=create_model()
# ------------------------------------------
# DEFINICIÓN DE LA RED NEURONAL r0
# ------------------------------------------

# model = keras.Sequential([
#    keras.layers.Dense(20, activation=tf.nn.tanh,input_shape=(N_entr)),
#    keras.layers.Dense(15, activation=tf.nn.tanh),
#    #keras.layers.Dense(50, activation=tf.nn.tanh), 
#    keras.layers.Dense(1, activation='sigmoid')
#])

# model.compile(loss=['mean_squared_error'],optimizer='adam',metrics=['mean_squared_error'])

# ------------------------------------------
# DEFINICIÓN DE LA RED NEURONAL r1
# ------------------------------------------

model = keras.Sequential([
    keras.layers.Dense(20, activation=tf.nn.relu,input_shape=(N_entr)),
    keras.layers.Dense(15, activation=tf.nn.relu),
    #keras.layers.Dense(50, activation=tf.nn.tanh), 
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=['mean_squared_error'],optimizer='Adam',metrics=['mean_squared_error'])



model.summary()




checkpoint_path = "training_"+str(savenum)+"/cp-{epoch:04d}.ckpt"


checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1,period=5)#one each5


start_time = time.time()

historial=model.fit(xx_training_set, yy_training_set,  epochs = 4000,batch_size=1024,
          validation_data = (xx_validation_set,yy_validation_set))
#help(historial)


    
elapsed_time = time.time() - start_time
print("EL PROCESO HA FINALIZADO. DURACION ",elapsed_time, "SEGUNDOS")

savenum=savenum+1
np.save("savenum",savenum)

#to storage the model at the end
model.save('./checkpoints/my_modell'+str(savenum)+'.h5') 


# In[21]:


#Predicting
yy_pred = model.predict(xx_validation_set)


savenum=np.load("savenum.npy")

print_result(yy_pred,yy_validation_set,savenum)
print_hist(historial,savenum)


# In[22]:


#30 % se indica como 0.3
clasific(yy_pred,yy_validation_set,0.1)
print(" ")
clasific(yy_pred,yy_validation_set,0.3)
print(" ")
clasific(yy_pred,yy_validation_set,0.5)


# In[23]:


# REPRESENTACIÓN DE LAS PREDICCIONES CON DATOS NORMALIZADOS

pred_NTU=(yy_pred*maximus[0]+minimun[0])
validation_NTU=(yy_validation_set*maximus[0]+minimun[0])


yy_pred = model.predict(xx_validation_set)


savenum=np.load("savenum.npy")

print_result(pred_NTU,validation_NTU,savenum)


# In[ ]:





# In[24]:


#TO load a full trained model
savenum=np.load("savenum.npy")
new_model = keras.models.load_model('./checkpoints/my_modell'+str(savenum)+'.h5')
yy_pred = new_model.predict(xx_validation_set)
err=yy_pred-yy_validation_set
print(err)


# In[ ]:




