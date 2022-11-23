from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS

#Librerias para manipulacion de datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Librerias para la conexion con la BD Oracle
import cx_Oracle

#Librerias para el manejos de fechas
from datetime import date
from datetime import datetime
from datetime import timedelta

#Librerias para el algoritmo de Recomendacion
# Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel
#Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#Import cosine_similarity function
from sklearn.metrics.pairwise import cosine_similarity

#Entrenar modelo KNN
from sklearn.preprocessing import StandardScaler                      #Standarizacion
from sklearn.model_selection import train_test_split                  #Obtencion de data de Training y Test
from sklearn.neighbors import KNeighborsClassifier                    #Modelo KNN
from sklearn.metrics import classification_report,confusion_matrix    #Matriz de confusion
import random

#Algoritmo SMOTE
from imblearn.over_sampling import SMOTE

#Validacion de modelos
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)
api = Api(app)

#Habilitar CORS (Consulta desde cualquier aplicacion externa a este programa)
CORS(app)

#FUNCIONES

#Manejo de fechas
def diferenciaDias(fecha1, fecha2):
    return (fecha2 - fecha1).days
 
def str_to_date(fecha):
    dia = int(fecha[8:10])
    mes = int(fecha[5:7])
    anio = int(fecha[:4])
    return date(anio, mes, dia)

#Tranforma a entero una variable
def to_int(x):
    try:
        x = pd.to_numeric(x)
    except:
        x = np.nan
    return x

#Obtener un dataFrame que solo contenga los elementos no duplicados
def obtenerDataframeSoloConElementosNoDuplicados(dfFull,dfFiltrar,dfFiltraOriginal,dfAComparar):

    #dfFull es la union de dfAComparar + dfFiltrar (en ese orden)
    
    #Encontrar elementos duplicados
    isDuplicado = dfFull.duplicated(dfFull.columns[dfFull.columns.isin(['nombre'])])
    isDuplicado
    
    #Obtener eventos nuevos que no estan registrados en la BD
    i=0
    countNoDuplicados = 0
    countDuplicado = 0
    for duplicado in isDuplicado:

        #Se eliminaran elementos de dfFiltrar que también esten en dfAComparar
        if i >= len(dfAComparar):
            #print (duplicado)
            if duplicado == False:
                print(i-len(dfAComparar),' - ',dfFiltraOriginal.iloc[i-len(dfAComparar)].nombre.encode("utf-8")," ---> no esta duplicado")
                countNoDuplicados = countNoDuplicados + 1 
            else:
                print(i-len(dfAComparar),' - ',dfFiltraOriginal.iloc[i-len(dfAComparar)].nombre.encode("utf-8")," ---> esta duplicado",len(dfFiltrar))
                dfFiltrar = dfFiltrar.drop([i-len(dfAComparar)],axis=0) #elimina el elemento i de dfFiltrar
                countDuplicado = countDuplicado + 1
        i=i+1

    #Finalmente solo se obtiene los elementos de dfFiltrar que no esten en dfAComparar
    print("countDuplicado:",countDuplicado,"countNoDuplicados:",countNoDuplicados,"dfFiltrar:",len(dfFiltrar))
    
    #Resetear indices
    dfFiltrar.index = range(dfFiltrar.shape[0])
    
    return dfFiltrar


def obtenerDataframeSoloConElementosNoDuplicadosIntereses(dfFull,dfFiltrar,dfFiltraOriginal,dfAComparar):
    
    #dfFull es la union de dfAComparar + dfFiltrar (en ese orden)
    
    #Encontrar elementos duplicados
    isDuplicado = dfFull.duplicated(dfFull.columns[dfFull.columns.isin(['usuario','evento'])])
    isDuplicado
    
    #Obtener eventos nuevos que no estan registrados en la BD
    i=0
    countNoDuplicados = 0
    countDuplicado = 0
    for duplicado in isDuplicado:

        #Se eliminaran elementos de dfFiltrar que también esten en dfAComparar
        if i >= len(dfAComparar):
            #print (duplicado)
            if duplicado == False:
                print(i-len(dfAComparar),' - ',dfFiltraOriginal.iloc[i-len(dfAComparar)].usuario,dfFiltraOriginal.iloc[i-len(dfAComparar)].evento.encode("utf-8")," ---> no esta duplicado")
                countNoDuplicados = countNoDuplicados + 1 
            else:
                print(i-len(dfAComparar),' - ',dfFiltraOriginal.iloc[i-len(dfAComparar)].usuario,dfFiltraOriginal.iloc[i-len(dfAComparar)].evento.encode("utf-8")," ---> esta duplicado",len(dfFiltrar))
                dfFiltrar = dfFiltrar.drop([i-len(dfAComparar)],axis=0) #elimina el elemento i de dfFiltrar
                countDuplicado = countDuplicado + 1
        i=i+1

    #Finalmente solo se obtiene los elementos de dfFiltrar que no esten en dfAComparar
    print("countDuplicado:",countDuplicado,"countNoDuplicados:",countNoDuplicados,"dfFiltrar:",len(dfFiltrar))
    
    #Resetear indices
    dfFiltrar.index = range(dfFiltrar.shape[0])
    
    return dfFiltrar

#Funcion para obtener el numero del mes
def obtenerMes(mesString):   
    switcher = {
        "enero": 1,
        "febrero": 2,
        "marzo": 3,
        "abril": 4,
        "mayo": 5,
        "junio": 6,
        "julio": 7,
        "agosto": 8,
        "setiembre": 9,
        "octubre": 10,
        "noviembre": 11,
        "diciembre": 12
    }
    return switcher.get(mesString, -1)

#Funcion para obtener el numero del mes
def obtenerMes2(mesString):   
    switcher = {
        "ene": 1,
        "feb": 2,
        "mar": 3,
        "abr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "ago": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dic": 12
    }
    return switcher.get(mesString, -1)

#Funcion para tranformar la fecha a un formato de fecha valida
def transformarFecha(fecha):
    fechaInicioArray = fecha.split('de')
    
    #Numero de sub partes de la fecha
    #print(len(fechaInicioArray))
    #for x in fechaInicioArray:
    #    print(x)
    
    if len(fechaInicioArray) == 3: #Jueves, 17  de  octubre  de  2019  - 21:00
        dia = fechaInicioArray[0].split(',')[1].strip()
        #print(dia)
        
        mes = obtenerMes(fechaInicioArray[1].strip().lower())
        #print(mes)
        
        agno = fechaInicioArray[2].split('-')[0].strip()
        #print(agno)
        
        hora= fechaInicioArray[2].split('-')[1].strip().split(':')[0]
        #print(hora)
        
        minuto= fechaInicioArray[2].split('-')[1].strip().split(':')[1]
        #print(minuto)
        
        fechaFormateada = datetime(int(agno), int(mes), int(dia), int(hora), int(minuto), 00, 00000)
        
        if int(hora) < 7:
            fechaFormateada = fechaFormateada + timedelta(days=1)
        
        #datetime(2019, 2, 28, 10, 15, 00, 00000)
        return fechaFormateada
    elif len(fechaInicioArray) == 2:  #30 de sep., 9:00
        dia = fechaInicioArray[0].strip()
        #print(dia)
        
        mes = obtenerMes2(fechaInicioArray[1].split('.,')[0].strip().lower())
        #print(mes)
        
        agno = date.today().year
        #print(agno)
        
        hora= fechaInicioArray[1].split('.,')[1].strip().split(':')[0]
        #print(hora)
        
        minuto= fechaInicioArray[1].split('.,')[1].strip().split(':')[1]
        #print(minuto)
        
        fechaFormateada = datetime(int(agno), int(mes), int(dia), int(hora), int(minuto), 00, 00000)
        
        #datetime(2019, 2, 28, 10, 15, 00, 00000)
        return fechaFormateada
    elif len(fechaInicioArray) == 1:  #29/09/2019 20:00:00 
        fechaInicioArrayTeleticket = fechaInicioArray[0].split('/')
        
        #print ("len:",len(fechaInicioArrayTeleticket))
        if len(fechaInicioArrayTeleticket) == 3:
            dia = fechaInicioArrayTeleticket[0].strip()
            #print(dia)

            mes = fechaInicioArrayTeleticket[1].strip()
            #print(mes)

            agno = fechaInicioArrayTeleticket[2].split(' ')[0].strip()
            #print(agno)

            hora= fechaInicioArrayTeleticket[2].split(' ')[1].strip().split(':')[0]
            #print(hora)

            minuto= fechaInicioArrayTeleticket[2].split(' ')[1].strip().split(':')[1]
            #print(minuto)

            fechaFormateada = datetime(int(agno), int(mes), int(dia), int(hora), int(minuto), 00, 00000)
            
            return fechaFormateada

#===============================================
#ALGORITMO DE FILTRADO POR CONTENIDO
#===============================================

#ALGORITMO DE RECOMENDACION UTILIZANDO SOLO EL CAMPO DE DESCRIPCION
def algoritmoRecomendacion_obtener_dataset_recomendacion(usuario):
    dfEventosRecomendacion = dfInteresesFull[(dfInteresesFull['usuario'] == usuario)]
    dfEventosRecomendacion.index = range(dfEventosRecomendacion.shape[0]) #Arreglar los indices
    
    #Limpieza de datos (lowercase)
    for feature in ['descripcion','evento', 'ubicacion', 'ubicacion_detalle', 'distrito','region','organizador','interes_eventos','interes_artistas','interes_generos','descripcionPersonal']:
        dfEventosRecomendacion[feature] = dfEventosRecomendacion[feature].apply(sanitize)
    
    return dfEventosRecomendacion

# Function to sanitize data to prevent ambiguity.
# Removes spaces and converts to lowercase
def sanitize(x):
    if isinstance(x, list):
        #Strip spaces and convert to lowercase
        #return [str.lower(i.replace(" ", "")) for i in x]
        return [str.lower(i) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
          #return str.lower(x.replace(" ", ""))
          return str.lower(x)
        else:
          return ''

def algoritmoRecomendacion_obtenerCosineSim(dfEventosRecomendacion):
    #Define a TF-IDF Vectorizer Object. Remove all english stopwords
    #tfidf = TfidfVectorizer(stop_words='spanish')  #No existe stop words para spanish
    tfidf = TfidfVectorizer()

    #Replace NaN with an empty string - UTILIZANDO SOLO EL CAMPO DE DESCRIPCION
    dfEventosRecomendacion['descripcion'] = dfEventosRecomendacion['descripcion'].fillna('')

    #Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
    tfidf_matrix = tfidf.fit_transform(dfEventosRecomendacion['descripcion'])

    #Output the shape of tfidf_matrix
    print("Dimensiones:",tfidf_matrix.shape)
    
    #Obtener cosine similarity score
    #Cual es la similitud que existe entre los todos eventos de musica (la diagonal es siempre 1)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

def algoritmoRecomendacion_indices(dfEventosRecomendacion):
    #Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any 
    indices = pd.Series(dfEventosRecomendacion.index, index=dfEventosRecomendacion['evento']).drop_duplicates()
    return indices

# Function that takes in movie title as input and gives recommendations
def algoritmoRecomendacion_filtrado_por_contenido_descripcion(title, cosine_sim, df, indices):
    # Obtain the index of the movie that matches the title
    idxs = indices[title]

    for idx in idxs:
      # Get the pairwsie similarity scores of all movies with that movie
      # And convert it into a list of tuples as described above
      sim_scores = list(enumerate(cosine_sim[idx]))
      # Sort the movies based on the cosine similarity scores
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      # Get the scores of the 10 most similar movies. Ignore the first event.
      sim_scores = sim_scores[1:11]
      print (sim_scores)
      # Get the movie indices
      event_indices = [i[0] for i in sim_scores]
      # Return the top 10 most similar movies
    return df['evento'].iloc[event_indices]


#ALGORITMO DE RECOMENDACION UTILIZANDO VARIOS CAMPOS
#Function that creates a soup out of the desired metadata
#LE DOY PESO 3 a GENEROS Y ARTISTAS
def create_soup(x):
    
    return ( ''.join(x['descripcion']) + ' ' + 
    ''.join(x['evento']) + ' ' + 
    ''.join(x['evento']) + ' ' + 
    #''.join(x['ubicacion']) + ' ' + 
    #''.join(x['ubicacion_detalle']) + ' ' + 
    ''.join(x['distrito']) + ' ' + 
    ''.join(x['region']) + ' ' + 
    ''.join(x['organizador']) + ' ' + 
    ''.join(x['interes_eventos']) + ' ' + 
    ''.join(x['interes_artistas']) + ' ' + 
    ''.join(x['interes_artistas']) + ' ' + 
    ''.join(x['interes_artistas']) + ' ' + 
    ''.join(x['interes_generos']) + ' ' + 
    ''.join(x['interes_generos']) + ' ' + 
    ''.join(x['interes_generos']) + ' ' + 
    ''.join(x['descripcionPersonal']) + ' ' + 
    ''.join(x['descripcionPersonal']) 
    )

def algoritmoRecomendacion_indices_obtener_dataset_recomendacion_multiples_campos(usuario):
    dfEventosRecomendacion = dfInteresesFull[(dfInteresesFull['usuario'] == usuario)]
    dfEventosRecomendacion.index = range(dfEventosRecomendacion.shape[0]) #Arreglar los indices
    
    #Limpieza de datos (lowercase)
    for feature in ['descripcion','evento', 'ubicacion', 'ubicacion_detalle', 'distrito','region','organizador','interes_eventos','interes_artistas','interes_generos','descripcionPersonal']:
        dfEventosRecomendacion[feature] = dfEventosRecomendacion[feature].apply(sanitize)
    
    # Create the new soup feature
    dfEventosRecomendacion['soup'] = dfEventosRecomendacion.apply(create_soup, axis=1)
    
    return dfEventosRecomendacion

# algoritmo recomendacion - CountVectorizer
def algoritmoRecomendacion_filtrado_por_contenido_multiples_campos_CountVectorizer(title, df):
        
    #Algoritmo CountVectorizer
    count = CountVectorizer()
    count_matrix = count.fit_transform(df['soup'])

    #Compute the cosine similarity score (equivalent to dot product for tf-idf vectors)
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Reset index of your df and construct reverse mapping again
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['evento'])
    
    # Obtain the index of the movie that matches the title
    idxs = indices[title]

    for idx in idxs:
      # Get the pairwsie similarity scores of all movies with that movie
      # And convert it into a list of tuples as described above
      sim_scores = list(enumerate(cosine_sim[idx]))
      # Sort the movies based on the cosine similarity scores
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      # Get the scores of the 10 most similar movies. Ignore the first event.
      sim_scores = sim_scores[1:11]
      print (sim_scores)
      # Get the movie indices
      event_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return df['evento'].iloc[event_indices]

# algoritmo de recomendacion - TfidfVectorizer
# algoritmo de recomendacion - TfidfVectorizer
def algoritmoRecomendacion_filtrado_por_contenido_multiples_campos_TfidfVectorizer(title, df, indicesEventosFavoritos):  
    
    
    #Algoritmo TfidfVectorizer
    count = TfidfVectorizer()
    count_matrix = count.fit_transform(df['soup'])

    #Compute the cosine similarity score (equivalent to dot product for tf-idf vectors)
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Reset index of your df and construct reverse mapping again
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['evento'])
    
    
    # Obtain the index of the movie that matches the title
    #global idxs
    idxs = indices[title]
    
    #Evaluar si solo se obtuvo 1 indice numerico, lo vuelve una lista
    if type(idxs) != list:
        lista = []
        lista.append(idxs)
        idxs = lista
    
    for idx in idxs:
        
        # Get the pairwsie similarity scores of all movies with that movie
        # And convert it into a list of tuples as described above
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the movies based on the cosine similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the 30 most similar movies. Ignore the first event.
        sim_scores = sim_scores[1:51]
        #print (sim_scores)
        # Get the movie indices
        event_indices = [i[0] for i in sim_scores]
        # Return the top 10 most similar movies
      
    #Eliminar indices de eventos que ya le gusta el usuario (se buscan nuevas recomendaciones)]
    for indEliminar in indicesEventosFavoritos:
    
        if indEliminar in event_indices:
            
            #Obtener los indices para luego eliminarla en sim_scores (Utilizamos los event_indices originales obtenidos de sim_scores)
            indiceEliminarSimScores = event_indices.index(indEliminar)
            
            #Eliminar elemento de event_indices (por valor) y de sim_scores (por indice)
            event_indices.remove(indEliminar)
            sim_scores.pop(indiceEliminarSimScores)
    
    #print("event_indices",event_indices)
    print("sim_scores",sim_scores)
    #print(indicesEventosFavoritos)
    return event_indices, sim_scores
    #return df['evento'].iloc[event_indices]

#FILTRADO COLABORATIVO
def obtenerMatrizValoracionUsuarios():
    lista = []

    #Obtenemos en una lista todos los usuarios del sistema
    for index, row in dfUsersBD.iterrows():
        lista.append(row.nombre)

    #Creamos un dataset con las columnas de los usuarios
    dfMatrizValoracionUsuarios = pd.DataFrame(columns= lista)

    #Añadimos a cada columna las valoraciones hechas por los usuarios
    for index, row in dfUsersBD.iterrows():
        dfMatrizValoracionUsuariosTemp = dfInteresUserEventBD[(dfInteresUserEventBD['usuario'] == row.nombre)]['valoracion']
        dfMatrizValoracionUsuariosTemp.index = range(dfMatrizValoracionUsuariosTemp.shape[0])
        dfMatrizValoracionUsuarios[row.nombre] = dfMatrizValoracionUsuariosTemp
        dfMatrizValoracionUsuarios[row.nombre] = dfMatrizValoracionUsuarios[row.nombre].apply(to_int)
        
    return dfMatrizValoracionUsuarios

def obtenerValorOptimoK(X_train, X_test, y_train, y_test):
    tasa_error = []

    # Tomará algún tiempo
    for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        tasa_error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),tasa_error,color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Tasa de Error vs. Valor de K')
    plt.xlabel('K')
    plt.ylabel('Tasa de Error')

#Entrenando los mejores hiperparametros de entrenamiento para el modelo KNN
def obtenerMejoresHiperParametrosModeloKNN(X_train,y_train):
    param_grid = { 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'] , 'n_neighbors' : [1,2,3,4,5,6,7,8], 'p':[1,2,3,4,5], 'leaf_size':[1,2,3,4,5,10,30]}

    #grid = GridSearchCV(KNeighborsClassifier(random_state=60),param_grid,refit=True,verbose=2,n_jobs =1)
    grid = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=2,n_jobs =1)
    grid.fit(X_train,y_train)
    
    print(grid.best_params_)

def ejecucionAlgoritmoFiltradoColaborativo(dfMatrizValoracionUsuarios,usuario,seed):

    #Estandarizar las variables
    scaler = StandardScaler()
    scaler.fit(dfMatrizValoracionUsuarios.drop(usuario,axis=1))
    scaled_features = scaler.transform(dfMatrizValoracionUsuarios.drop(usuario,axis=1))

    #Obteniendo data de entrenamiento y Test (Semilla: seed)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features,dfMatrizValoracionUsuarios[usuario],
                                                        test_size=0.30, random_state=seed)  

    #Vemos que la data esta desbalanceada
    print("\nData desbalanceada: ")
    print(dfMatrizValoracionUsuarios[usuario].value_counts())

    #Balanceamos la data a predecir - algoritmo SMOTE
    sm = SMOTE(random_state=seed, k_neighbors = 1)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print("\nData balanceada: ")
    print(np.bincount(y_train))

    #Entrenamos el modelo escogido
    knn = KNeighborsClassifier(n_neighbors= 1)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)

    #Matriz de confusion
    print("\nMatriz de Confusion: ")
    print(confusion_matrix(y_test,pred))

    #Metricas de evaluacion
    print("\nMetricas de evaluacion: ")
    print(classification_report(y_test,pred))
    
    #Método del codo: obtener el mejor valor de K
    obtenerValorOptimoK(X_train, X_test, y_train, y_test)
    
    #Obtener los mejores hiperParametros
    #obtenerMejoresHiperParametrosModeloKNN(X_train,y_train)
    
    #Obtener la precision del modelo
    scoring='accuracy'

    #Spot Check Algorithms
    models=[]
    models.append(('LR',LogisticRegression()))
    models.append(('LDA',LinearDiscriminantAnalysis()))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('CART',DecisionTreeClassifier()))
    models.append(('NB',GaussianNB()))
    models.append(('SVM',SVC()))

    #evaluate each model in turn
    results=[]
    names=[]
    print("\nValidación cruzada: Comparativa con los demás modelos: ")
    for name,model in models:
        kfold=model_selection.KFold(n_splits=10,random_state=seed)
        cv_results=model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg="Modelo %s - Precisión: %f (Desviación estándar: %f)"%(name,cv_results.mean(),cv_results.std())
        print(msg)

class AlgoritmoRecomendacion(Resource):
    #Connect as user "USR_SREM" with password "USR_SREM" to the "orcl" service running on this computer.
    global connection
    global cursor
    connection = cx_Oracle.connect("USR_SREM", "USR_SREM", "localhost:1521/orcl", encoding="UTF-8")
    cursor = connection.cursor()

    #========================================
    #OBTENCION DE DATOS BD
    #========================================
    
    #---------------------------------------------------------
    #EVENTOS
    #Creamos los dataframes para los eventos recolectados en BD
    global dfEventosBD
    dfEventosBD = pd.DataFrame(columns=['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region','fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'])
    dfEventosBDTemp = pd.DataFrame(columns=['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region','fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'])

    #Añadir un registro al dataframe para que despues dicho registro sea modificado e insertado en un nuevo dataframe
    dfEventosBDTemp = dfEventosBDTemp.append({'nombre': 'temp'}, ignore_index=True)

    #Obtener la información almacenada en la BD de los eventos
    #Parametros Busqueda
    sentenciaSQL = """
      SELECT EVEN_NOMBR, EVEN_UBICA, EVEN_UBIDE, EVEN_UBIRE, EVEN_DISTR, EVEN_REGIO, EVEN_FYHIN, EVEN_FYHIA, EVEN_FYHFI, EVEN_DESCR, EVEN_PRECIO, EVEN_ORGAN, EVEN_ASIST, EVEN_INTER, EVEN_COMPA, EVEN_IMAGE, EVEN_LINKE from SREMEVENTO ORDER BY EVEN_ID"""

    cursor.execute(sentenciaSQL)
    for nombre,ubicacion,ubicacion_detalle,ubicacion_referencia,distrito,region,fecha_inicio,fecha_inicio_alt,fecha_fin,descripcion,precio,organizador,asistiran,me_interesa,veces_compartido,imagen,enlace_evento in cursor:
     
      dfEventosBDTemp['nombre'] = nombre                      
      dfEventosBDTemp['ubicacion'] = ubicacion                
      dfEventosBDTemp['ubicacion_detalle'] = ubicacion_detalle    
      dfEventosBDTemp['ubicacion_referencia'] = ubicacion_referencia              
      dfEventosBDTemp['distrito'] =  distrito
      dfEventosBDTemp['region'] =  region
      dfEventosBDTemp['fecha_inicio'] =  fecha_inicio
      dfEventosBDTemp['fecha_inicio_alt'] =  fecha_inicio_alt
      dfEventosBDTemp['fecha_fin'] =  fecha_fin
      dfEventosBDTemp['descripcion'] = ( descripcion.read() if descripcion != None else '') #Leer valores CLOB
      dfEventosBDTemp['precio'] =  precio
      dfEventosBDTemp['organizador'] =  organizador
      dfEventosBDTemp['asistiran'] =  asistiran
      dfEventosBDTemp['me_interesa'] =  me_interesa
      dfEventosBDTemp['veces_compartido'] =  veces_compartido
      dfEventosBDTemp['imagen'] =  imagen
      dfEventosBDTemp['enlace_evento'] =  enlace_evento
    
      dfEventosBD = pd.concat([dfEventosBD, dfEventosBDTemp])
    dfEventosBD.index = range(dfEventosBD.shape[0]) #Arreglar los indices

    #Regularizacion de la data vacia y del tipo de dato
    dfEventosBD = dfEventosBD.fillna('')

    dfEventosBD['asistiran'] = dfEventosBD['asistiran'].apply(to_int)
    dfEventosBD['me_interesa'] = dfEventosBD['me_interesa'].apply(to_int)
    dfEventosBD['veces_compartido'] = dfEventosBD['veces_compartido'].apply(to_int)

    dfEventosBD['asistiran'] = dfEventosBD['asistiran'].fillna(-1)
    dfEventosBD['me_interesa'] = dfEventosBD['me_interesa'].fillna(-1)
    dfEventosBD['veces_compartido'] = dfEventosBD['veces_compartido'].fillna(-1)

    

    def get(self,nick):

      #========================================
      #OBTENCION DE DATOS BD
      #========================================

      #-------------------------------------------------------------------------------
      #INTERES DE LOS USUARIOS
      #Creamos los dataframes para los intereses de los usuarios recolectados en BD
      global dfInteresUserEventBD
      dfInteresUserEventBD = pd.DataFrame(columns=['usuario','evento','interes','veces_visitado','valoracion'])
      dfInteresUserEventBDTemp = pd.DataFrame(columns=['usuario','evento','interes','veces_visitado','valoracion'])

      #Añadir un registro al dataframe para que despues dicho registro sea modificado e insertado en un nuevo dataframe
      dfInteresUserEventBDTemp = dfInteresUserEventBDTemp.append({'usuario': 'temp'}, ignore_index=True)

      #Obtener la información almacenada en la BD de los intereses de los usuarios por los eventos
      #Parametros Busqueda
      sentenciaSQL = """
      SELECT USUA_NICKN, EVEN_NOMBR, USIN_INTER, USIN_VISIT, USIN_VALOR FROM SREMUSUAIN ORDER BY USIN_ID"""

      cursor.execute(sentenciaSQL)
      for usuario,evento,interes,veces_visitado,valoracion in cursor:
       
        dfInteresUserEventBDTemp['usuario'] = usuario                      
        dfInteresUserEventBDTemp['evento'] = evento                
        dfInteresUserEventBDTemp['interes'] = interes    
        dfInteresUserEventBDTemp['veces_visitado'] = veces_visitado              
        dfInteresUserEventBDTemp['valoracion'] =  valoracion
      
        dfInteresUserEventBD = pd.concat([dfInteresUserEventBD, dfInteresUserEventBDTemp])
      dfInteresUserEventBD.index = range(dfInteresUserEventBD.shape[0]) #Arreglar los indices
      

      #---------------------------------------------------------
      #USUARIOS
      #Creamos los dataframes para los usuarios recolectados en BD
      global dfUsersBD
      dfUsersBD = pd.DataFrame(columns=['nombre','interes_eventos','interes_artistas','interes_generos','descripcionPersonal'])
      dfUsersBDTemp = pd.DataFrame(columns=['nombre','interes_eventos','interes_artistas','interes_generos','descripcionPersonal'])

      #Añadir un registro al dataframe para que despues dicho registro sea modificado e insertado en un nuevo dataframe
      dfUsersBDTemp = dfUsersBDTemp.append({'nombre': 'temp'}, ignore_index=True)

      #Obtener la información almacenada en la BD de los usuario

      #Parametros Busqueda
      sentenciaSQL = """
      SELECT USUA_NICKN, USUA_EVENT, USUA_MUSIC, USUA_GENER, USUA_DESCR FROM SREMUSUARI ORDER BY USUA_ID"""

      cursor.execute(sentenciaSQL)
      for nombre,interes_eventos,interes_artistas,interes_generos,descripcionPersonal in cursor:
       
        dfUsersBDTemp['nombre'] = nombre                      
        dfUsersBDTemp['interes_eventos'] =  ( interes_eventos.read() if interes_eventos != None else '')               
        dfUsersBDTemp['interes_artistas'] =  ( interes_artistas.read() if interes_artistas != None else '')   
        dfUsersBDTemp['interes_generos'] =  ( interes_generos.read() if interes_generos != None else '')               
        dfUsersBDTemp['descripcionPersonal'] =   ( descripcionPersonal.read() if descripcionPersonal != None else '')
    
        dfUsersBD = pd.concat([dfUsersBD, dfUsersBDTemp])
      dfUsersBD.index = range(dfUsersBD.shape[0]) #Arreglar los indices 
      

      #Regularizacion de data
      dfUsersBD = dfUsersBD.fillna('')

      #======================================================
      #CREAMOS EL DATASET COMPLETO PARA LAS RECOMENDACIONES
      #======================================================
      
      global dfInteresesFull
      dfInteresesFull = dfInteresUserEventBD

      #Añadiendo campos de los eventos
      dfInteresesFull = pd.merge(dfInteresesFull, dfEventosBD, left_on='evento', right_on='nombre', how='outer')

      #Añadiendo campos de los usuario
      dfInteresesFull = pd.merge(dfInteresesFull, dfUsersBD, left_on='usuario', right_on='nombre', how='outer')

      #Regularizacion de los datos
      dfInteresesFull = dfInteresesFull.fillna('')
      dfInteresesFull['interes'] = dfInteresesFull['interes'].apply(to_int)
      dfInteresesFull['veces_visitado'] = dfInteresesFull['veces_visitado'].apply(to_int)
      dfInteresesFull['valoracion'] = dfInteresesFull['valoracion'].apply(to_int)
      dfInteresesFull['asistiran'] = dfInteresesFull['asistiran'].apply(to_int)
      dfInteresesFull['me_interesa'] = dfInteresesFull['me_interesa'].apply(to_int)
      dfInteresesFull['veces_compartido'] = dfInteresesFull['veces_compartido'].apply(to_int)

      #meanMargen = df['Margen'].mean()
      dfInteresesFull['asistiran'] = dfInteresesFull['asistiran'].fillna(-1)
      dfInteresesFull['me_interesa'] = dfInteresesFull['me_interesa'].fillna(-1)
      dfInteresesFull['veces_compartido'] = dfInteresesFull['veces_compartido'].fillna(-1)

      #========================================================
      #ALGORITMO DE RECOMENDACION UTILIZANDO MULTIPLES CAMPOS
      #========================================================
      dfEventosRecomendacionCamposultiples = algoritmoRecomendacion_indices_obtener_dataset_recomendacion_multiples_campos(nick)

      #Obtener los eventos que le gustan al usuario (los que tienen valoracion mayor a 1)
      dfEventosFavoritosUsuario = dfEventosRecomendacionCamposultiples.sort_values(['valoracion','me_interesa'], 
        ascending=[False,False])[dfEventosRecomendacionCamposultiples.valoracion > 1]

      #Lista de indices obtenitdos de las recomendaciones
      listaGeneralIndicesRecomendacion = []

      #Iteramos cada uno de los eventos que ya le gustan al usuario
      for indice_fila in dfEventosFavoritosUsuario.index:

          print("*****************************************************")
          print("indice_fila",indice_fila)
          #obtener los sim_scores de cada uno de los eventos
          indicesRecomendacion, sim_scores = algoritmoRecomendacion_filtrado_por_contenido_multiples_campos_TfidfVectorizer(indice_fila,dfEventosRecomendacionCamposultiples,dfEventosFavoritosUsuario.index)
          
          #Añadir los sim_scores obtenidos de todas las recomendaciones a la lista de indices
          for i in range(len(sim_scores)):
              listaGeneralIndicesRecomendacion.append(sim_scores[i])

      #Agrugar las recomendaciones en base al indice (en caso que halla mas de un registro con el mismo indice => se agrupara como media)
      #Creamos una dataframe con la lista de indices obtenidos
      dfObjTemp = pd.DataFrame(listaGeneralIndicesRecomendacion) 
      dfObjTemp.columns = ['indice','probabilidad']

      #Agrupamos por los indices obtenidos de la recomendacion
      dfAgrupado = dfObjTemp.groupby(['indice']).mean()                         #la probabilidad obtenida sera la media
      dfAgrupado['conteo'] = dfObjTemp.groupby(['indice']).count()['probabilidad']  #contar las veces que se repite un indice


      # Ordenar por los valores de las columna 'conteo' y 'probabilidad':
      #Obtenemos los primeros 10 elementos
      #dfAgrupado = dfAgrupado.sort_values(by=['conteo','probabilidad'], ascending=[False,False])[:10]
      #dfAgrupado = dfAgrupado.sort_values(by=['probabilidad','conteo'], ascending=[False,False])[:10]
      
      #Obtnemos todos los elementos
      dfAgrupado = dfAgrupado.sort_values(by=['probabilidad','conteo'], ascending=[False,False])
      print(dfAgrupado)

      #Obtenemos los indices del dataFrame de indices ya agrupado y filtrado
      indicesRecomendacion =  [int(ind) for ind in dfAgrupado.index]    #Es necesario transformarlo a tipo de dato int nativo de Python para que sea serializable
      indicesRecomendacionOriginal =  [int(ind) for ind in dfAgrupado.index]
      nombreEventos = []

      contEventosRecomendados = 0  #Conteo de eventos recomendados que pasaron el filtro de fecha / tambien sera utilizado para eliminar indices

      for x in indicesRecomendacionOriginal:
        print ("=========================================================")
        print("indicesRecomendacionOriginal:",indicesRecomendacionOriginal)
        print("indicesRecomendacion:",indicesRecomendacion)
        print("contEventosRecomendados:",contEventosRecomendados)
        #Solo se obtendra los 10 primeros eventos
        #if contEventosRecomendados == 10:
        #  break

        #fechaInicioEvento = dfEventosRecomendacionCamposultiples.iloc[x].fecha_inicio
        fechaInicioEvento =  str(dfEventosRecomendacionCamposultiples.iloc[x].fecha_inicio)

        #Validar si el evento tiene el campo fecha
        if len(fechaInicioEvento) > 0:
            fechaHoy = date.today()
            fechaEvento = str_to_date(fechaInicioEvento)
            result = diferenciaDias(fechaHoy, fechaEvento)
            print ("fecha hoy:",fechaHoy,"fecha evento:",fechaEvento)
            if result >= 0:
                print ("Faltan",result,"días para el evento",dfEventosRecomendacionCamposultiples.iloc[x].evento.encode("utf-8"))
                nombreEventos.append(dfEventosRecomendacionCamposultiples.iloc[x].evento.encode("utf-8"))
                contEventosRecomendados = contEventosRecomendados + 1
            elif result < 0:
                print ("El evento ",dfEventosRecomendacionCamposultiples.iloc[x].evento.encode("utf-8")," es pasado :'(")
                #Eliminar elemento de los indices de eventos recomendados
                #del indicesRecomendacion[contEventosRecomendados]
        else:
            print ("Lamentablemente el evento ",dfEventosRecomendacionCamposultiples.iloc[x].evento.encode("utf-8"),"no se posee fecha registrada")
            #Eliminar elemento de los indices de eventos recomendados
            del indicesRecomendacion[contEventosRecomendados]

      #========================================
      #FILTRADO COLABORATIVO
      #========================================
      
      dfMatrizValoracionUsuarios = obtenerMatrizValoracionUsuarios()
      print("\nMatriz interes usuario-evento:")
      print(dfMatrizValoracionUsuarios)

      #Calculamos la matriz de correlacion de Pearson
      print("\nCalculamos la matriz de correlacion de Pearson")
      print(dfMatrizValoracionUsuarios.corr(method ='pearson'))
      plt.matshow(dfMatrizValoracionUsuarios.corr())

      #Obtenemos la validacion y comparativa del modelo
      ejecucionAlgoritmoFiltradoColaborativo(dfMatrizValoracionUsuarios,nick,60)


      print('nombreEventos',nombreEventos)
      return {'indicesRecomendacion': indicesRecomendacion, 'nombreEventos': nombreEventos, 'nick': nick}



class ActualizarBDInteresesNuevoUsuario(Resource):

    #Connect as user "USR_SREM" with password "USR_SREM" to the "orcl" service running on this computer.
    global connection
    global cursor
    connection = cx_Oracle.connect("USR_SREM", "USR_SREM", "localhost:1521/orcl", encoding="UTF-8")
    cursor = connection.cursor()

    #========================================
    #OBTENCION DE DATOS BD
    #========================================
    
    #---------------------------------------------------------
    #EVENTOS
    #Creamos los dataframes para los eventos recolectados en BD
    global dfEventosBD
    dfEventosBD = pd.DataFrame(columns=['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region','fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'])
    dfEventosBDTemp = pd.DataFrame(columns=['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region','fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'])

    #Añadir un registro al dataframe para que despues dicho registro sea modificado e insertado en un nuevo dataframe
    dfEventosBDTemp = dfEventosBDTemp.append({'nombre': 'temp'}, ignore_index=True)

    #Obtener la información almacenada en la BD de los eventos
    #Parametros Busqueda
    sentenciaSQL = """
      SELECT EVEN_NOMBR, EVEN_UBICA, EVEN_UBIDE, EVEN_UBIRE, EVEN_DISTR, EVEN_REGIO, EVEN_FYHIN, EVEN_FYHIA, EVEN_FYHFI, EVEN_DESCR, EVEN_PRECIO, EVEN_ORGAN, EVEN_ASIST, EVEN_INTER, EVEN_COMPA, EVEN_IMAGE, EVEN_LINKE from SREMEVENTO ORDER BY EVEN_ID"""

    cursor.execute(sentenciaSQL)
    for nombre,ubicacion,ubicacion_detalle,ubicacion_referencia,distrito,region,fecha_inicio,fecha_inicio_alt,fecha_fin,descripcion,precio,organizador,asistiran,me_interesa,veces_compartido,imagen,enlace_evento in cursor:
     
      dfEventosBDTemp['nombre'] = nombre                      
      dfEventosBDTemp['ubicacion'] = ubicacion                
      dfEventosBDTemp['ubicacion_detalle'] = ubicacion_detalle    
      dfEventosBDTemp['ubicacion_referencia'] = ubicacion_referencia              
      dfEventosBDTemp['distrito'] =  distrito
      dfEventosBDTemp['region'] =  region
      dfEventosBDTemp['fecha_inicio'] =  fecha_inicio
      dfEventosBDTemp['fecha_inicio_alt'] =  fecha_inicio_alt
      dfEventosBDTemp['fecha_fin'] =  fecha_fin
      dfEventosBDTemp['descripcion'] = ( descripcion.read() if descripcion != None else '') #Leer valores CLOB
      dfEventosBDTemp['precio'] =  precio
      dfEventosBDTemp['organizador'] =  organizador
      dfEventosBDTemp['asistiran'] =  asistiran
      dfEventosBDTemp['me_interesa'] =  me_interesa
      dfEventosBDTemp['veces_compartido'] =  veces_compartido
      dfEventosBDTemp['imagen'] =  imagen
      dfEventosBDTemp['enlace_evento'] =  enlace_evento
    
      dfEventosBD = pd.concat([dfEventosBD, dfEventosBDTemp])
    dfEventosBD.index = range(dfEventosBD.shape[0]) #Arreglar los indices

    #Regularizacion de la data vacia y del tipo de dato
    dfEventosBD = dfEventosBD.fillna('')

    dfEventosBD['asistiran'] = dfEventosBD['asistiran'].apply(to_int)
    dfEventosBD['me_interesa'] = dfEventosBD['me_interesa'].apply(to_int)
    dfEventosBD['veces_compartido'] = dfEventosBD['veces_compartido'].apply(to_int)

    dfEventosBD['asistiran'] = dfEventosBD['asistiran'].fillna(-1)
    dfEventosBD['me_interesa'] = dfEventosBD['me_interesa'].fillna(-1)
    dfEventosBD['veces_compartido'] = dfEventosBD['veces_compartido'].fillna(-1)

    def get(self,nick_user):

      #Creamos un dataSet Temporal para añadir los intereses de los usuarios para usuarios nuevos
      dfTemp = pd.DataFrame(columns=['usuario','evento','interes','veces_visitado','valoracion'])

      dfTemp['evento'] = dfEventosBD[['nombre']]    #evento
      dfTemp['interes'] = 0                #interes
      dfTemp['veces_visitado'] = 0         #veces_visitado
      dfTemp['valoracion'] = 0             #valoracion 
      dfTemp['usuario'] = nick_user   #nombre de usuario

      dfTemp.index = range(dfTemp.shape[0]) #Arreglar los indices

      #================================================================
      #PERSISTENCIA EN BD
      #================================================================
      #INTERES USUARIOS
    
      #Insertar registro
      sentenciaInsertSQL = """
      INSERT INTO SREMUSUAIN(USIN_ID,USUA_ID,EVEN_ID,
      USUA_NICKN,EVEN_NOMBR,USIN_INTER,USIN_VISIT,USIN_VALOR) 
      values (SREMUSUAIN_SEQ.NEXTVAL,:usua_id,:even_id,:usuario,
      :evento,:interes,:veces_visitado,:valoracion)"""

      sentenciaUsuarioId = """
      SELECT USUA_ID FROM SREMUSUARI WHERE USUA_NICKN = :usuario """

      sentenciaEventoId = """
      SELECT EVEN_ID FROM SREMEVENTO WHERE EVEN_NOMBR = :evento """

      usuarioId = ""
      eventoId = ""

      for index, row in dfTemp.iterrows():
        
        cursor.execute(sentenciaUsuarioId,  usuario = row['usuario'])
        for usua_id in cursor:
            usuarioId = usua_id
        usuarioId = usuarioId[0]
    
        cursor.execute(sentenciaEventoId,  evento = row['evento'])
        for even_id in cursor:
            eventoId = even_id
        eventoId = eventoId[0]
    
        print (row['usuario'], ' - ', row['evento'].encode("utf-8") , ' - ', usuarioId, ' - ', eventoId)
    
        cursor.execute(sentenciaInsertSQL, 
                   usua_id = usuarioId, even_id = eventoId, usuario = row['usuario'],
                   evento = row['evento'], interes = row['interes'],
                   veces_visitado = row['veces_visitado'],
                   valoracion = row['valoracion']
                  )
                   #region = row['region'],fecha_inicio = new_date,fecha_inicio_alt = new_date,
                   #fecha_fin = new_date)
        connection.commit()

      return {'intereses-nuevo-usuario': 'OK'}

class ActualizarBD(Resource):
    def get(self):
      #Importamos los .csv obtenidos de Scrapy
      dfFB = pd.read_csv('EVENTS_FB.csv', sep='|')
      dfTeleticket = pd.read_csv('EVENTS_TELETICKET.csv', sep='|')

      #===============================
      #OBTENER INFORMACION DE LA BD
      #===============================

      #-------------------------------------------------------------------------------------------
      #TABLA - INTERESES DE USUARIOS
      #Creamos los dataframes para los intereses de los usuarios recolectados en BD
      dfInteresUserEventBD = pd.DataFrame(columns=['usuario','evento','interes','veces_visitado','valoracion'])
      dfInteresUserEventBDTemp = pd.DataFrame(columns=['usuario','evento','interes','veces_visitado','valoracion'])

      #Añadir un registro al dataframe para que despues dicho registro sea modificado e insertado en un nuevo dataframe
      dfInteresUserEventBDTemp = dfInteresUserEventBDTemp.append({'usuario': 'temp'}, ignore_index=True)

      # Connect as user "USR_SREM" with password "USR_SREM" to the "orcl" service running on this computer.
      connection = cx_Oracle.connect("USR_SREM", "USR_SREM", "localhost:1521/orcl", encoding="UTF-8")
      cursor = connection.cursor()

      #Parametros Busqueda
      sentenciaSQL = """
      SELECT USUA_NICKN, EVEN_NOMBR, USIN_INTER, USIN_VISIT, USIN_VALOR FROM SREMUSUAIN ORDER BY USIN_ID"""

      cursor.execute(sentenciaSQL)
      for usuario,evento,interes,veces_visitado,valoracion in cursor:
         
        dfInteresUserEventBDTemp['usuario'] = usuario                      
        dfInteresUserEventBDTemp['evento'] = evento                
        dfInteresUserEventBDTemp['interes'] = interes    
        dfInteresUserEventBDTemp['veces_visitado'] = veces_visitado              
        dfInteresUserEventBDTemp['valoracion'] =  valoracion
      
        dfInteresUserEventBD = pd.concat([dfInteresUserEventBD, dfInteresUserEventBDTemp])
      dfInteresUserEventBD.index = range(dfInteresUserEventBD.shape[0]) #Arreglar los indices

      #---------------------------------------------------------------------------------------------
      #TABLA - USUARIOS
      #Creamos los dataframes para los usuarios recolectados en BD
      dfUsersBD = pd.DataFrame(columns=['nombre','interes_eventos','interes_artistas','interes_generos','descripcionPersonal'])
      dfUsersBDTemp = pd.DataFrame(columns=['nombre','interes_eventos','interes_artistas','interes_generos','descripcionPersonal'])

      #Añadir un registro al dataframe para que despues dicho registro sea modificado e insertado en un nuevo dataframe
      dfUsersBDTemp = dfUsersBDTemp.append({'nombre': 'temp'}, ignore_index=True)

      # Connect as user "USR_SREM" with password "USR_SREM" to the "orcl" service running on this computer.
      connection = cx_Oracle.connect("USR_SREM", "USR_SREM", "localhost:1521/orcl", encoding="UTF-8")
      cursor = connection.cursor()

      #Parametros Busqueda
      sentenciaSQL = """
      SELECT USUA_NICKN, USUA_EVENT, USUA_MUSIC, USUA_GENER, USUA_DESCR FROM SREMUSUARI ORDER BY USUA_ID"""

      cursor.execute(sentenciaSQL)
      for nombre,interes_eventos,interes_artistas,interes_generos,descripcionPersonal in cursor:
        #print("********************************")
        #print("Values:", nombre, ubicacion, ubicacion_detalle, ubicacion_referencia,distrito)
         
        dfUsersBDTemp['nombre'] = nombre                      
        dfUsersBDTemp['interes_eventos'] =  ( interes_eventos.read() if interes_eventos != None else '')               
        dfUsersBDTemp['interes_artistas'] =  ( interes_artistas.read() if interes_artistas != None else '')   
        dfUsersBDTemp['interes_generos'] =  ( interes_generos.read() if interes_generos != None else '')               
        dfUsersBDTemp['descripcionPersonal'] =   ( descripcionPersonal.read() if descripcionPersonal != None else '')
        
        dfUsersBD = pd.concat([dfUsersBD, dfUsersBDTemp])
      dfUsersBD.index = range(dfUsersBD.shape[0]) #Arreglar los indices 

      #--------------------------------------------------------------------------------------
      #TABLA - EVENTOS

      #Creamos los dataframes para los eventos recolectados en BD
      dfEventosBD = pd.DataFrame(columns=['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region','fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'])
      dfEventosBDTemp = pd.DataFrame(columns=['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region','fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'])

      #Añadir un registro al dataframe para que despues dicho registro sea modificado e insertado en un nuevo dataframe
      dfEventosBDTemp = dfEventosBDTemp.append({'nombre': 'temp'}, ignore_index=True)

      # Connect as user "USR_SREM" with password "USR_SREM" to the "orcl" service running on this computer.
      connection = cx_Oracle.connect("USR_SREM", "USR_SREM", "localhost:1521/orcl", encoding="UTF-8")
      cursor = connection.cursor()

      #Parametros Busqueda
      sentenciaSQL = """
      SELECT EVEN_NOMBR, EVEN_UBICA, EVEN_UBIDE, EVEN_UBIRE, EVEN_DISTR, EVEN_REGIO, EVEN_FYHIN, EVEN_FYHIA, EVEN_FYHFI, EVEN_DESCR, EVEN_PRECIO, EVEN_ORGAN, EVEN_ASIST, EVEN_INTER, EVEN_COMPA, EVEN_IMAGE, EVEN_LINKE from SREMEVENTO ORDER BY EVEN_ID"""

      cursor.execute(sentenciaSQL)
      for nombre,ubicacion,ubicacion_detalle,ubicacion_referencia,distrito,region,fecha_inicio,fecha_inicio_alt,fecha_fin,descripcion,precio,organizador,asistiran,me_interesa,veces_compartido,imagen,enlace_evento in cursor:
        #print("********************************")
        #print("Values:", nombre, ubicacion, ubicacion_detalle, ubicacion_referencia,distrito)
         
        dfEventosBDTemp['nombre'] = nombre                      
        dfEventosBDTemp['ubicacion'] = ubicacion                
        dfEventosBDTemp['ubicacion_detalle'] = ubicacion_detalle    
        dfEventosBDTemp['ubicacion_referencia'] = ubicacion_referencia              
        dfEventosBDTemp['distrito'] =  distrito
        dfEventosBDTemp['region'] =  region
        dfEventosBDTemp['fecha_inicio'] =  fecha_inicio
        dfEventosBDTemp['fecha_inicio_alt'] =  fecha_inicio_alt
        dfEventosBDTemp['fecha_fin'] =  fecha_fin
        dfEventosBDTemp['descripcion'] = ( descripcion.read() if descripcion != None else '')
        dfEventosBDTemp['precio'] =  precio
        dfEventosBDTemp['organizador'] =  organizador
        dfEventosBDTemp['asistiran'] =  asistiran
        dfEventosBDTemp['me_interesa'] =  me_interesa
        dfEventosBDTemp['veces_compartido'] =  veces_compartido
        dfEventosBDTemp['imagen'] =  imagen
        dfEventosBDTemp['enlace_evento'] =  enlace_evento
        
        dfEventosBD = pd.concat([dfEventosBD, dfEventosBDTemp])

      dfEventosBD.index = range(dfEventosBD.shape[0]) #Arreglar los indices

      #========================================
      #Manipulación de Data
      #========================================
  
      #---------------------------------------------------------------------
      #EVENTOS
      #Eliminar eventos duplicados por la columna nombre
      dfFB = dfFB.drop_duplicates(dfFB.columns[dfFB.columns.isin(['nombre'])], keep='first')
      dfTeleticket = dfTeleticket.drop_duplicates(dfTeleticket.columns[dfTeleticket.columns.isin(['nombre'])], keep='first')

      #UNIMOS LOS DATAFRAMES DE LOS EVENTOS DE FACEBOOK Y TELETICKET
      dfEventos = dfFB.append(dfTeleticket)
      dfEventos.index = range(dfEventos.shape[0]) #Arreglar los indices

      #Regularizacion de la data vacia y del tipo de dato
      dfEventos = dfEventos.fillna('')

      dfEventos['asistiran'] = dfEventos['asistiran'].apply(to_int)
      dfEventos['me_interesa'] = dfEventos['me_interesa'].apply(to_int)
      dfEventos['veces_compartido'] = dfEventos['veces_compartido'].apply(to_int)

      dfEventos['asistiran'] = dfEventos['asistiran'].fillna(-1)
      dfEventos['me_interesa'] = dfEventos['me_interesa'].fillna(-1)
      dfEventos['veces_compartido'] = dfEventos['veces_compartido'].fillna(-1)

      dfEventosBD = dfEventosBD.fillna('')

      dfEventosBD['asistiran'] = dfEventosBD['asistiran'].apply(to_int)
      dfEventosBD['me_interesa'] = dfEventosBD['me_interesa'].apply(to_int)
      dfEventosBD['veces_compartido'] = dfEventosBD['veces_compartido'].apply(to_int)

      dfUsersBD = dfUsersBD.fillna('')

      #Integramos los dataframes de eventos de la BD y de los archivos recolectados en Scrapy
      dfEventosFullIntegrada = pd.DataFrame(columns=['nombre','ubicacion','ubicacion_detalle','ubicacion_referencia','distrito','region','fecha_inicio','fecha_inicio_alt','fecha_fin','descripcion','precio','organizador','asistiran','me_interesa','veces_compartido','imagen','enlace_evento'])
      dfEventosNuevosSinBD = dfEventos


      #UNIMOS LOS DATAFRAMES DE LOS EVENTOS DE SCRAPY Y DE BD
      dfEventosFullIntegrada = dfEventosBD.append(dfEventos)
      dfEventosFullIntegrada.index = range(dfEventosFullIntegrada.shape[0]) #Arreglar los indices

      #-------------------------------------------------------------
      #INTERESES DE USUARIO
      #Creando el dataFrame de Intereses de los usuarios
      dfIntereses = pd.DataFrame(columns=['usuario','evento','interes','veces_visitado','valoracion'])
      dfTemp = pd.DataFrame(columns=['usuario','evento','interes','veces_visitado','valoracion'])

      for index, row in dfUsersBD.iterrows():
        #print (row['nombre'])   
        
        dfTemp['evento'] = dfEventosFullIntegrada[['nombre']]    #evento
        dfTemp['interes'] = 0                #interes
        dfTemp['veces_visitado'] = 0         #veces_visitado
        dfTemp['valoracion'] = 0             #valoracion 
        dfTemp['usuario'] =  row['nombre']   #nombre de usuario
        dfIntereses = pd.concat([dfIntereses, dfTemp])
        dfIntereses.index = range(dfIntereses.shape[0]) #Arreglar los indices

      #Integramos los dataframes de los intereses de la BD y de los instereses construidos
      dfInteresUserEventIntegrada = pd.DataFrame(columns=['usuario','evento','interes','veces_visitado','valoracion'])
      dfInteresUserEventSinBD = dfIntereses

      dfInteresUserEventIntegrada = dfInteresUserEventBD.append(dfIntereses) #pd.concat([dfInteresUserEventBD, dfIntereses])
      dfInteresUserEventIntegrada.index = range(dfInteresUserEventIntegrada.shape[0]) #Arreglar los indices
      dfInteresUserEventIntegrada

      #==============================================
      #OBTENCION DE DATASET PARA GUARDARLAS EN BD
      #==============================================
      #Obtener un dataFrame que solo contenga los elementos nuevos que no estan duplicados
      #Eventos
      dfEventosNuevosSinBD = obtenerDataframeSoloConElementosNoDuplicados(dfEventosFullIntegrada,dfEventosNuevosSinBD,dfEventos,dfEventosBD)
      #Interes usuarios
      dfInteresUserEventSinBD = obtenerDataframeSoloConElementosNoDuplicadosIntereses(dfInteresUserEventIntegrada,dfInteresUserEventSinBD,dfIntereses,dfInteresUserEventBD)

      #==============================================
      #PERSISTENCIA EN BD
      #==============================================

      #--------------------------------------------------------------------
      #EVENTOS
      # Connect as user "USR_SREM" with password "USR_SREM" to the "orcl" service running on this computer.
      connection = cx_Oracle.connect("USR_SREM", "USR_SREM", "localhost:1521/orcl", encoding="UTF-8")
      cursor = connection.cursor()
      
      #Insertar registro
      sentenciaInsertSQL = """INSERT INTO 
      SREMEVENTO(EVEN_ID,EVEN_NOMBR,EVEN_UBICA,EVEN_UBIDE,
      EVEN_UBIRE,EVEN_DISTR,EVEN_REGIO,EVEN_FYHIN,EVEN_FYHIA,
      EVEN_FYHFI,EVEN_DESCR,EVEN_PRECIO,EVEN_ORGAN,EVEN_ASIST,
      EVEN_INTER,EVEN_COMPA,EVEN_VALOR,EVEN_IMAGE,EVEN_LINKE) 
      VALUES (SREMEVENTO_SEQ.NEXTVAL,:nombre,:ubicacion,
      :ubicacion_detalle,:ubicacion_referencia,:distrito,:region,
      :fecha_inicio,:fecha_inicio_alt,:fecha_fin,:descripcion,:precio,
      :organizador,:asistiran,:me_interesa,:veces_compartido,
      :valoracion,:imagen_evento,:enlace_evento)"""


      for index, row in dfEventosNuevosSinBD.iterrows():
        print (row['nombre'], ' - ', row['fecha_inicio'] , ' - ' , row['fecha_inicio_alt'] , ' - ', row['fecha_fin'])
         
        fechaInicio = transformarFecha(row['fecha_inicio'])
        print("fechaInicio:",fechaInicio)
        
        fechaInicioAlt = transformarFecha(row['fecha_inicio_alt'])
        print("fechaInicioAlt:",fechaInicioAlt)
        
        fechaFin = transformarFecha(row['fecha_fin'])
        print("fechaFin:",fechaFin)
        
        cursor.execute(sentenciaInsertSQL, 
                     nombre = row['nombre'], ubicacion = row['ubicacion'], ubicacion_detalle = row['ubicacion_detalle'],
                     ubicacion_referencia = row['ubicacion_referencia'], distrito = row['distrito'],
                     region = row['region'],fecha_inicio = fechaInicio, fecha_inicio_alt = fechaInicioAlt,
                     fecha_fin = fechaFin,descripcion = row['descripcion'],precio = row['precio'],
                     organizador = row['organizador'],asistiran = row['asistiran'],me_interesa = row['me_interesa'],
                     veces_compartido = row['veces_compartido'],valoracion = 0,
                     imagen_evento = row['imagen'],enlace_evento = row['enlace_evento']
                    )
                     #region = row['region'],fecha_inicio = new_date,fecha_inicio_alt = new_date,
                     #fecha_fin = new_date)
        connection.commit()

      #INTERES USUARIOS
      # Connect as user "USR_SREM" with password "USR_SREM" to the "orcl" service running on this computer.
      connection = cx_Oracle.connect("USR_SREM", "USR_SREM", "localhost:1521/orcl", encoding="UTF-8")
      cursor = connection.cursor()
    
      #Insertar registro
      sentenciaInsertSQL = """
      INSERT INTO SREMUSUAIN(USIN_ID,USUA_ID,EVEN_ID,
      USUA_NICKN,EVEN_NOMBR,USIN_INTER,USIN_VISIT,USIN_VALOR) 
      values (SREMUSUAIN_SEQ.NEXTVAL,:usua_id,:even_id,:usuario,
      :evento,:interes,:veces_visitado,:valoracion)"""

      sentenciaUsuarioId = """
      SELECT USUA_ID FROM SREMUSUARI WHERE USUA_NICKN = :usuario """

      sentenciaEventoId = """
      SELECT EVEN_ID FROM SREMEVENTO WHERE EVEN_NOMBR = :evento """

      usuarioId = ""
      eventoId = ""

      for index, row in dfInteresUserEventSinBD.iterrows():
        
        cursor.execute(sentenciaUsuarioId,  usuario = row['usuario'])
        for usua_id in cursor:
            usuarioId = usua_id
        usuarioId = usuarioId[0]
    
        cursor.execute(sentenciaEventoId,  evento = row['evento'])
        for even_id in cursor:
            eventoId = even_id
        eventoId = eventoId[0]
    
        print (row['usuario'], ' - ', row['evento'].encode("utf-8") , ' - ', usuarioId, ' - ', eventoId)
    
        cursor.execute(sentenciaInsertSQL, 
                   usua_id = usuarioId, even_id = eventoId, usuario = row['usuario'],
                   evento = row['evento'], interes = row['interes'],
                   veces_visitado = row['veces_visitado'],
                   valoracion = row['valoracion']
                  )
                   #region = row['region'],fecha_inicio = new_date,fecha_inicio_alt = new_date,
                   #fecha_fin = new_date)
        connection.commit()

      return {'actualizarBD': 'OK'}

api.add_resource(AlgoritmoRecomendacion, '/obtenerRecomendacion/<nick>')
api.add_resource(ActualizarBDInteresesNuevoUsuario, '/intereses-nuevo-usuario/<nick_user>')
api.add_resource(ActualizarBD, '/')

if __name__ == '__main__':
    app.run(debug=True)
