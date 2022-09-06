#!/usr/bin/env python
# coding: utf-8

# # **Proyecto final Data Science - Coderhouse - Juan Ignacio Szurlewicz** 

# ## Título: Wheather prediction in Seattle

# El objetivo del presente proyecto es obtener mediante un set de datos de la ciudad de Seattle y un algoritmo de clasificación, un modelo que nos permita pronosticar de manera categórica el clima futuro, contemplando un conjunto de variables observadas.Título: Wheather prediction on Seattle

# El set de datos empleado se encuentra en http://kaggle.com, en el siguiente link :
# https://www.kaggle.com/datasets/ananthr1/weather-prediction

# # Data Acquisition

# Para la realización de la etapa de Data Acquisition se procedió a la búsqueda de un dataset que a travéz de variables claves, nos permita llegar a nuestra variable target propuesta. En este sentido, nos encontramos con el dataset de variables climatologicas de la ciudad de Seattle. El mismo, se encontraba en formato .CSV, es decir, de texto en columnas, clasificandose como un tipo de dato estructurado.

# Para la lectura del archivo utilizamos la libreria pandas que, a travéz del comando read_csv nos permite la carga de este tipo de archivos planos.

# In[355]:


import pandas as pd
import datetime as dt


# In[356]:


_df=pd.read_csv(r"C:\Users\Novix\Downloads\seattle-weather.csv")


# In[357]:


_df.head()


# Así, pudimos completar la lectura de nuestro dataset y a través del comando Head obtener los primeros 5 valores del conjunto de datos y observar, cuáles son nuestras variables consideradas. El siguiente paso es ir hacia el Data Wrangling, mediante el cuál haremos un descubrimiento mayor del conjunto de datos, su limpieza, estructuración y validación.

# # Data Wrangling

# La etapa del datawrangling consiste en la manipulación de los datos obtenidos en la etapa previa, de forma tal que sean más amigables y adecuados para su trabajo y para el logro del objetivo final. Dentro de esta etapa, iremos abarcando las distintas tareas del descubrimiento, estructuración, limpieza, enriquecimiento, validación y publicación.

# Comenzamos con la lectura del dataframe:

# In[358]:


_df.head() #Aquí identificamos a primera vista las variables que componen el set de datos


# In[359]:


_df.shape #Vemos el tamaño de nuestro dataset


# In[360]:


_df.isna().sum() #Analizamos si tenemos algun missing value en el set de datos


# In[361]:


_df.describe().T  #Obtenemos una visión general de las variables, con sus principales estadísticas descriptivas


# In[362]:


_df.dtypes #Aquí podemos ver los tipos de datos, vamos a pasar la variables date a datetime. El resto están bien asignadas


# In[363]:


_df['date'] = pd.to_datetime(_df['date'])


# In[364]:


_df.dtypes #Quedaron los tipos corregidos


# # Exploratory Data Analysis

# El exploratory data analysis consiste en una etapa que tiene como finalidad la exploración del conjunto de variables, de forma univariada como así tambien de forma bivariada y multivariada. Debemos comprender que representa cada una de ellas, si existen casos atípicos que puedan influir la consistencia de nuestro modelo antes de aplicar el mismo. Se deben preparar los datos para que los mismos representen de la forma más fiel el eje del problema en cuestión. Para ello, se realizan analisis gráficos en una primera instancia y luego analisis de medidas descriptivas. Si es necesario, se deben decidir cursos de acción para solucionar problemas como la presencia de missing values y valores outliers.

# In[365]:


#Importamos las librerias necesarias
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from seaborn import boxplot
from seaborn import violinplot
import numpy as np


# ### Análisis univariado de las variables consideradas

# #### 1) Wheater

# In[366]:


_df.weather.describe()


# In[367]:


sns.countplot(x=_df['weather'],palette='Spectral')


# In[368]:


frec = _df["weather"].value_counts()
Wheater_freq=pd.DataFrame(frec)
Wheater_freq=Wheater_freq.rename(columns={'weather':'Frecuencia Absoluta'})
Frec_abs_val = Wheater_freq["Frecuencia Absoluta"].values
acum = []
valor_acum = 0
for i in Frec_abs_val:
    valor_acum = valor_acum + i
    acum.append(valor_acum)
    
Wheater_freq["frec_abs_acum"] = acum
Wheater_freq["frec_rel_%"] = round(100 * Wheater_freq["Frecuencia Absoluta"]/len(_df.weather),4)
Frec_rel_val = Wheater_freq["frec_rel_%"].values
acum = []
valor_acum = 0
for i in Frec_rel_val:
    valor_acum = valor_acum + i
    acum.append(valor_acum)
    
Wheater_freq["frec_rel_%_acum"] = acum
Wheater_freq


# Por medio de la tabla de frecuencias podemos ver que entre la condición climática "lluvia" y "soleado" se presenta en casi 90% de los casos.

# #### 2) Precipitation

# Esta variable se encuentra en pulgadas

# In[369]:


_df.precipitation.describe().T


# In[370]:


boxplot(x=_df.precipitation, orient="Vertical")


# Como podemos observar, encontramos valores outliers superiores, pero lo atribuimos a la naturaleza de la variable precipitación, ya que se trata de un fenómeno natural que puede presentar eventos disruptivos que no siguen una distribución normal.

# In[371]:


sns.histplot(data=_df.precipitation,kde=True)
plt.xlim(0,50)


# Podemos ver una clara asimetria hacia la derecha

# In[372]:


frec = _df["precipitation"].value_counts()
precipitation_freq=pd.DataFrame(frec)
precipitation_freq=precipitation_freq.rename(columns={'precipitation':'Frecuencia Absoluta'})
Frec_abs_val = precipitation_freq["Frecuencia Absoluta"].values
acum = []
valor_acum = 0
for i in Frec_abs_val:
    valor_acum = valor_acum + i
    acum.append(valor_acum)
    
precipitation_freq["frec_abs_acum"] = acum
precipitation_freq["frec_rel_%"] = round(100 * precipitation_freq["Frecuencia Absoluta"]/len(_df.precipitation),4)
Frec_rel_val = precipitation_freq["frec_rel_%"].values
acum = []
valor_acum = 0
for i in Frec_rel_val:
    valor_acum = valor_acum + i
    acum.append(valor_acum)
    
precipitation_freq["frec_rel_%_acum"] = acum
precipitation_freq.sort_values(by='Frecuencia Absoluta',ascending=False)


# In[373]:


precipitation_freq.head()  #Los primeros 5 valores en la distribución de frecuencia absoluta


# In[374]:


precipitation_freq.tail()   #Los últimos 5 valores en la distribución de frecuencia absoluta


# ### 3) Wind

# Esta variable se encuentra en millas por hora Mph

# In[375]:


_df.wind.describe()


# In[376]:


sns.histplot(data=_df.wind,kde=True)
plt.xlim(0,10)


# En este caso podemos ver que la distribución es prácticamente normal no estandar

# In[377]:


violinplot(x=_df.wind, data=_df , orient="Verical")


# Aquí podemos visualizar mejor la distribución de los datos, dónde el ancho de la figura indica la frecuencia de ocurrencia absoluta de los valores.

# ### 3) Temp_Max

# Esta variable se encuentra en grados celcius C°

# In[378]:


_df.temp_max.describe()


# In[379]:


sns.histplot(data=_df.temp_max,kde=True)
plt.xlim(0,36)


# En este caso tambien, la distribución es prácticamente normal, media 16.43 y desviación estándar +- 7.35, si observamos las estadísticas descriptivas, el 25% de la distribución se da en 10.6 (16.43-7.35=9) y el 75% es en 22.2 (16.43+7.35=23.78)

# ### 3) Temp_Min

# Esta variable se encuentra en grados celcius C°

# In[380]:


_df.temp_min.describe()


# In[381]:


sns.histplot(data=_df.temp_min,kde=True)
plt.xlim(0,18.3)


# Aquí la distribución ya no parece la de una normal. La temperatura mínima observada se encuentra distribuida de manera más uniforme

# In[382]:


violinplot(x=_df.temp_min, data=_df , orient="Verical")


# La densidad que representa la figura es más amplia respecto al eje x, con lo cuál, para cada temperatura mínima existe un número similar de ocurrencias

# ### Análisis bivariado de las variables consideradas

# In[383]:


plt.rcParams['figure.figsize'] = (10,8)

sns.heatmap(_df.corr(), annot = True, cmap = 'Wistia')
plt.title('Mapa de correlaciones', fontsize = 20)
plt.show()


# Analizamos mediante el mapa de correlaciones la relación lineal entre las variables cuantitativas bajo análisis.

# ### 1) Weather vs Wind

# In[384]:


sns.scatterplot(x="weather", y="wind", data=_df)
plt.title('Relacion entre Sex and target', fontsize = 20, fontweight = 30)


# Podemos ver que en el caso de la llovizna, el viento es débil. Luego, para el caso de las lluvias, la distribución es mas uniforme. Por otro lado, en el caso del clima soleado, suele darse con mayor concentración para velocidades de viento menores. Por último, para el caso de la nieve y la niebla, hay algunos puntos pero la relación es débil. La niebla es más probable con poco viento.

# In[385]:


_df.groupby('weather')[['wind']].mean()  #Medias dentro de cada tipo de clima


# ### 2) Weather vs Precipitation

# In[386]:


sns.scatterplot(x="weather", y="precipitation", data=_df)
plt.title('Relacion entre Sex and target', fontsize = 20, fontweight = 30)


# Aquí podemos observar que las precipitaciones sólo se dan en el clima de lluvia y nieve. No se detectan precipitaciones en climas de neblina o llovizna.

# In[387]:


_df.groupby('weather')[['precipitation']].mean()  #Medias dentro de cada tipo de clima


# ### 3) Weather vs Temp_min

# In[388]:


sns.scatterplot(x="weather", y="temp_min", data=_df)
plt.title('Relacion entre Sex and target', fontsize = 20, fontweight = 30)


# Podemos observar de forma gráfica que los climas de llovizna, lluvia y niebla con compatibles con temperaturas mínimas mas elevadas. En el caso del clima soleado se presente para todas las temperaturas mínimas observadas y en el caso de nieve, para casos de temperaturas mínimas bajas.

# In[389]:


_df.groupby('weather')[['temp_min']].mean()  #Medias dentro de cada tipo de clima, reforzamos el análisis gráfico.v


# ### 3) Weather vs Temp_max

# In[390]:


sns.scatterplot(x="weather", y="temp_max", data=_df)
plt.title('Relacion entre Sex and target', fontsize = 20, fontweight = 30)


# Podemos observar de forma gráfica que los climas de llovizna, lluvia y niebla con compatibles con temperaturas máximas mas elevadas. En el caso del clima soleado se presente para todas las temperaturas mínimas observadas y en el caso de nieve, para casos de temperaturas mínimas bajas.

# In[391]:


_df.groupby('weather')[['temp_max']].mean()  #Medias dentro de cada tipo de clima


# ### Análisis multivariado de las variables consideradas

# In[392]:


sns.FacetGrid(_df,hue = 'weather' , height = 5).map(plt.scatter,'temp_min','temp_max').add_legend();
plt.show()


# In[393]:


sns.FacetGrid(_df,hue = 'weather' , height = 5).map(plt.scatter,'precipitation','wind').add_legend();
plt.show()


# In[394]:


sns.FacetGrid(_df,hue = 'weather' , height = 5).map(plt.scatter,'wind','temp_max').add_legend();
plt.show()


# In[395]:


sns.FacetGrid(_df,hue = 'weather' , height = 5).map(plt.scatter,'wind','temp_min').add_legend();
plt.show()


# In[396]:


sns.FacetGrid(_df,hue = 'weather' , height = 5).map(plt.scatter,'precipitation','temp_min').add_legend();
plt.show()


# In[397]:


sns.FacetGrid(_df,hue = 'weather' , height = 5).map(plt.scatter,'precipitation','temp_max').add_legend();
plt.show()


# Podemos observar que existe sólo un claro patrón en el primer caso, de temperatura min vs temperatura máx. En el resto de los casos, no hay un patrón.

# In[398]:


plt.figure(dpi=120)
sns.pairplot(_df)
plt.show()


# Aquí podemos observar lo mismo que antes y sacar las mismas conclusiones, sin discriminar por tipo de clima.

# In[399]:


_df.groupby('weather')[['precipitation','wind','temp_min','temp_max']].mean().T


# Mediante la tabla descriptiva podemos analizar que:
# 
# - **precipitation**: se presenta sólo en el clima de lluvia y nive, con una media de mm mayor en el último estado.
# - **wind**: se presenta en todos los estados, con mayor velocidad durante la nieve y climas soleados.
# - **temp_min**: se presenta en todos los estados
# - **temp_max**: se presenta en todos los estados

# In[400]:


_df['date']=pd.to_datetime(_df['date'])


# In[401]:


_df['date'] =_df['date'].dt.strftime('%m')


# In[402]:


_df.groupby('date')[['precipitation','wind','temp_min','temp_max']].mean().T


# Mediante dicha tabla podemos analizar el mes, tomando un promedio de los años analizados, con mayor media de cada variable. Entonces, Noviembre es el mes en el cuál mas lluvia hay, Diciembre es el mes con mas vientos, las temperaturas mínimas son mayores en Enero y las máximas en Agosto.

# ## Objetivo

# El objetivo principal del presente proyecto es obtener mediante un algoritmo de machine learning, un modelo que a travéz de los inputs analizados nos permita pronosticar el clima en la ciudad de Seattle. Nuestra variable target es entonces la variable "Weather" y las variables explicativas son las precipitaciones en pulgadas, la velocidad del viento en millas por hora y la temperatura mínima y máxima en grados celcius.

# In[ ]:




