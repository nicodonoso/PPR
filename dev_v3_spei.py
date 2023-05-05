#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:20:17 2023

@author: nico nicolasdonosocastro@gmai.com
0- recortar el netcdf con cdo:  cdo -sellonlatbox,-80.0,-50.0,-60.0,-10.0 /home/nico/Documentos/PPR/pdsi.mon.mean.nc /home/nico/Documentos/PPR/pdsi_cut.nc
1- leer netcdf
2- leer puntos de árboles
3- calcular los PCA
4- operar sobre la grilla
"""


import xarray as xr 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Input 

DIST  = 100# [km] la distancia del radio de búsqueda de las cronologías 
MINS  = 1# el numero mínimo de series que uno desea que incluya la reconstruccion de cada punto de grilla. La información del numero de series aparece en el archivo mismo de cada cronología.
MINFY = 1970#La fecha minima del primer año de la cronología para ser incluida en la reconstruccion. Es decir si la cronología cubre un periodo que va del 1900 al 2020 no podría incluirse en la reconstruccion. 
MINC  =  1#?Numero minimo de cronologias necesarias para incluir como predictores de la variable climatica a reconstruir. 
#ARMOD: Modelo autoregresivo 
NLAG = 3# [años] Permite incluir en el modelo la influencia de años previos. Igual no recuerdo bien si es que solo incluye los lags cuando la regresión es significativa.   
#SCREEN: Esta es una opción para filtrar aquellas cronologías que no están relacionadas con la variable climática. Si le indico que el screen sea de 90, solo me va a dejar las cronologías relacionadas significativamente con el clima al 90%. 
#NTAIL: evaluar la significancia a dos cola de la curva.
#SCALE: No recuerdo que significa. Tengo que chequearlo en mis anotaciones   
#SAVE:  Tampoco lo recuerdo, obviamente es para guardar algo
#WGT: Este es un parametro importante porque permite darle peso a las cronologías que mejor se relacionan con la variable climática.  
#HPLP: Creo que te permite aplicar un filtro de alta o baja frecuencia. No lo use nunca
#NBOOT: Nunca use el boot, ni se para que serviria.
'''
DIST=100#units in [km]
MINS: el numero mínimo de series que uno desea que incluya la reconstruccion de cada punto de grilla. La información del numero de series aparece en el archivo mismo de cada cronología.
MINFY: La fecha minima del primer año de la cronología para ser incluida en la reconstruccion. Es decir si la cronología cubre un periodo que va del 1900 al 2020 no podría incluirse en la reconstruccion. 
MINC: Numero minimo de cronologias necesarias para incluir como predictores de la variable climatica a reconstruir. 
ARMOD: Modelo autoregresivo 
NLAG: Permite incluir en el modelo la influencia de años previos. Igual no recuerdo bien si es que solo incluye los lags cuando la regresión es significativa.   
SCREEN: Esta es una opción para filtrar aquellas cronologías que no están relacionadas con la variable climática. Si le indico que el screen sea de 90, solo me va a dejar las cronologías relacionadas significativamente con el clima al 90%. 
NTAIL: evaluar la significancia a dos cola de la curva.
SCALE: No recuerdo que significa. Tengo que chequearlo en mis anotaciones   
SAVE:  Tampoco lo recuerdo, obviamente es para guardar algo
WGT: Este es un parametro importante porque permite darle peso a las cronologías que mejor se relacionan con la variable climática.  
HPLP: Creo que te permite aplicar un filtro de alta o baja frecuencia. No lo use nunca
NBOOT: Nunca use el boot, ni se para que serviria.
'''

#%% Leer netcdf
dire = '/home/nico/Documentos/PPR/'
file = 'pdsi_cut.nc'
#file = 'spei01.nc'
netcdf =  dire+file
ds = xr.open_dataset(netcdf)  # NetCDF or OPeNDAP URL
#%% Leer los datos de árboles, aquí apliqué un sed -i -e 's/,/./g' a ambos archivos para cambiar , por .
path_dir = '/home/nico/Documentos/PPR/'
std_file = 'std_cronos_PCA_25cronos_07072022_EPSrestriction.csv'
std_dendro = pd.read_csv(path_dir + std_file, sep =';',index_col=False)
std_dendro.rename(columns = {"yrs":"time"} , inplace = True)
std_dendro["time"] = pd.to_datetime(std_dendro["time"], format='%Y',errors = 'coerce')
std_dendro.dropna(subset = ["time"], inplace = True)#Elimino nos NaN que están en el "time" ¿es necesario?
#std_dendro.set_index("time",inplace = True)

#leo las coordenadas
coord_file = 'LoadingsPCA_22chronos_08072022.csv' 
qth = pd.read_csv(path_dir + coord_file, sep =';',index_col=0)

tiempo = 1630
#tiempo = 1450
ds_tiempo = ds.isel(time = tiempo)
pdsi = ds_tiempo['pdsi']

plt.figure()
pdsi.plot()
plt.plot(qth['Lon'],qth['Lat'],'r+')
#%% obtengo las latitudes y longitudes del xarray
latitud_field = ds.lat.values
longitd_field = ds.lon.values

for i in ds.lat.values:
    for j in ds.lon.values:
        #value = ds.sel(lat=i, lon=j)
        #print(value)
        #latitud = ds.lat.sel(lat=ds.lat[i]).values
        #print(ds.lat.sel(lat=ds.lat[i]).values)
        #longitud = ds.lon.sel(lon=ds.lon[j]).values
        print("latitud : " +str(i)+" longitud : "+str(j))
        #np.array([latitud,longitud])
#%% Busqueda de distancias entre los puntos del grillado y la ubicación de los árboles
#La distancia se calcula utilizando la fórmula del haversine, que es una aproximación que toma en cuenta la curvatura de la Tierra.
from geopy.distance import distance

coordenadas = np.array([qth['Lat'],qth['Lon']])
coordenadas = np.transpose(coordenadas)
# Seleccionamos el primer punto como referencia
ref_coord = coordenadas[0]

# Creamos una lista para almacenar las distancias
distancias_km = []

# Iteramos a través de los pares de coordenadas y calculamos la distancia entre cada par
for i in range(len(coordenadas)):
    coord = coordenadas[i]
    distancia_km = distance(ref_coord, coord).km
    distancias_km.append(distancia_km)
    print(f"La distancia entre {ref_coord} y {coord} es de {distancia_km:.2f} km")


#SELECCIONAMOS UN PUNTO DE GRILLA EN BASE A UN PUNTO DE QTH ÁRBOLES
latitud = qth['Lat'][0] #-33.87
longitud = qth['Lon'][0] #-71.42

TS = ds.sel(lon = longitud, lat = latitud, method='nearest')
plt.figure()
TS['pdsi'].plot()

pdsi = TS['pdsi']
pdsi = pdsi.to_pandas()
pdsi = pdsi.to_frame()
pdsi.rename(columns = {0:"pdsi"} , inplace = True)
pdsi.reset_index(inplace = True)
#pdsi['time']
#NECESITO AGRUPAR (IE. PROMEDIAR) EL INDICE PDSI PARA DEJARLO EN VALORES ANUALES

##pp1 = pdsi.groupby(pdsi.time.dt.year)['pdsi'].transform('mean') #-> Funciona pero me repite los valores por cada año
pdsi_y = pdsi.groupby(pdsi.time.dt.year)['pdsi'].mean()#calculo el promedio anual
pdsi_y = pdsi_y.to_frame()
pdsi_y.reset_index(inplace = True)
pdsi_y.columns = ['year', 'pdsi']

#preparo hacer el merge de los dos dataframe
std_dendro['year'] = std_dendro.time.dt.year #-> podria agregar esta columna 

df = std_dendro.merge(pdsi_y)
#Eliminamos datos de tiempos
df.drop(columns = ['time','year'],inplace = True)


#%% Relleno datos faltantes para calcular el PCA
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

imp_mean.fit(df)
std_dendro_impute = imp_mean.transform(df)

df = pd.DataFrame(std_dendro_impute,columns=df.columns)# Vuelvo a generar el df pero ahora no tienen NaN 
#Esto rellena los datos de dendro con los promedios, realmente necesito seleccionar por periodos disponibles

#%% Parte el preproceso para el cáculo de los PCA
X = df.drop('pdsi',axis = 1)
y = df['pdsi']
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#%% As mentioned earlier, PCA performs best with a normalized feature set. 
# We will perform standard scalar normalization to normalize our feature set. 
#To do this, execute the following code:

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()#transformar las características (variables) de tal manera que tengan una media igual a cero y una desviación estándar igual a uno.
X_train = sc.fit_transform(X_train)# ajustar el scaler y transformar los datos al mismo tiempo
X_test = sc.transform(X_test)#transformar los datos de entrenamiento y prueba utilizando los parámetros ajustados
# Why we use fit_transform() on training data but transform() on the test data?
#https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe

'''
Performing PCA using Scikit-Learn is a two-step process:

    Initialize the PCA class by passing the number of components to the constructor.
    Call the fit and then transform methods by passing the feature set to these methods. The transform method returns the specified number of principal components.
'''
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

pca = PCA()
# Ajusta el modelo PCA a los datos
pca.fit(X_train)
# Obtén las componentes principales
componentes_principales = pca.transform(X_train)
# Crea un modelo lineal utilizando solo el primer componente principal (PCA 1)
pca1 = componentes_principales[:,0].reshape(-1, 1)

y = y_train
# Crea un modelo lineal utilizando solo el primer componente principal (PCA 1)
modelo = LinearRegression().fit(pca1, y)




X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
print('Explained variance:')
print(explained_variance)

# Genera un gráfico con el resultado
fig, ax = plt.subplots()
ax.scatter(pca1, y)
ax.plot(pca1, modelo.predict(pca1), color='red')
ax.set_xlabel('PCA 1')
ax.set_ylabel('Variable de respuesta')
ax.set_title('Modelo lineal con PCA 1')
plt.text(-6,-6,'Varianza explicada: {:.1f}'.format(explained_variance[0]),fontsize=12)
plt.show()
#%% Preparamos los datos para el PCA






# #import random as rd
# #rd.seed()

# x = std_dendro.iloc[:,1:25]
# #preparamos el target
# #target = std_dendro.iloc[:,4]# debería ser pdsi cercano

# #Applying it to PCA function
# mat_reduced = PCA(x , 2)
 
# #Creating a Pandas DataFrame of reduced Dataset
# principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])


# def PCA(X , num_components):
     
#     #Step-1
#     X_meaned = X - np.mean(X , axis = 0)
     
#     #Step-2
#     cov_mat = np.cov(X_meaned , rowvar = False)
     
#     #Step-3
#     eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
#     #Step-4
#     sorted_index = np.argsort(eigen_values)[::-1]
#     sorted_eigenvalue = eigen_values[sorted_index]
#     sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
#     #Step-5
#     eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
#     #Step-6
#     X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
#     return X_reduced
