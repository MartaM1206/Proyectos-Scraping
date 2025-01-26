# Para evitar errores en la codificación forzamos a que sea utf-8
import sys
import os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Importamos librerías. De Beautiful Soup solo importamos una de sus clases
import requests
from bs4 import BeautifulSoup

# Definimos la función para extraer datos de la wikipedia
def extraer_datos_wikipedia(url):
    """Extrae datos de la Wikipedia.

    Args:
        url (string): url de la página de Wikipedia.
    """
    # Necesitamos un header para que Wikipedia no bloquee la solicitud. Esto se tiene que poner siempre igual.
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36"
    }
    # Hacemos una request de tipo get y le pasamos la url y el header para que se comporte como un agente, un navegador
    response = requests.get(url, headers=headers)
    # Si la respuesta no es 200 (200=Ok), entonces se imprime un mensaje con el código del error y se retorna None
    if response.status_code != 200:
        print(f"Error en la solicitud: {response.status_code}")
        return None
    
    # Lo siguente es parsear los datos de la respuesta, es decir, descomponer los datos para que sean comprensibles 
    # y poder usarlos y anlizarlos.
    parseo_a_html = BeautifulSoup(response.content, "html.parser")

    # Manejar contenido principal de forma segura
    contenido_principal = parseo_a_html.find("div", id="mw-content-text") 
    # El identificador id="mw-content-text" en principio siempre existe en wikipedia, aunque podría cambiar; 
    # en otras webs ese id cambia cada cierto tiempo
    contenido_texto = contenido_principal.getText() if contenido_principal else "Contenido principal no encontrado."

    # Extraer tablas
    # Buscamos todas las tablas que tengan la clase "wikitable"
    tablas = parseo_a_html.findAll("table", {"class": "wikitable"})
    # Creamos una lista vacía para guardar los datos de todas las tablas
    tablas_datos = []
    for tabla in tablas:
        # buscamos todas las filas de cada tabla encontrada y metida en la lista tablas_datos (tr = table row)
        filas = tabla.findAll("tr")
        # creamos una lista vacía para guardar los datos de cada tabla
        tabla_datos = []
        for fila in filas:
            # buscamos todas las celdas de cada fila encontrada (th = table header, td = table data)
            celdas = fila.findAll(["th", "td"])
            # extraemos el texto de cada celda y lo guardamos en una lista
            fila_datos = [celda.getText().strip() for celda in celdas]
            # añadimos la lista de datos de la fila a la lista de datos de la tabla
            tabla_datos.append(fila_datos)
        # añadimos la lista de datos de la tabla a la lista de datos de todas las tablas    
        tablas_datos.append(tabla_datos)
    # Conseguimos que cada tabla sea una lista de listas, donde cada lista es una fila de la tabla.
    # fila_datos podría ser la fila de encabezados de la tabla, por lo que sería la primera fila de la fila tabla_datos y, 
    # de ser la primera tabla encontrada, sería la primera tabla de la lista tablas_datos.


    # Con la información que hemos sacado creamos nuestro diccionario de datos
    #datos = {
    #"titulo": parseo_a_html.find("title").getText() if parseo_a_html.find("title") else "Título no encontrado.",
    #"enlaces": [a.getText() for a in parseo_a_html.findAll("a")[:10] if a.getText()],
    #"imagenes": [img.get("src") for img in parseo_a_html.findAll("img")[:5] if img.get("src")],
    #"texto": contenido_texto.replace('\u200b', ''),
    #"tablas": tablas_datos}

    datos = {"titulo": parseo_a_html.find("title").getText() if parseo_a_html.find("title") else "Título no encontrado.",
             "tablas": tablas_datos}
    return datos

# Llamamos a la función
datos_HP = extraer_datos_wikipedia("https://es.wikipedia.org/wiki/Harry_Potter_(serie_cinematogr%C3%A1fica)")
if extraer_datos_wikipedia:
    print("Título de la página:", datos_HP["titulo"])
    #print("Primeros 10 enlaces:", datos_HP["enlaces"])
    #print("Primeras 5 imágenes:", datos_HP["imagenes"])
    #print("Contenido:", datos_HP["texto"])
    print("Tablas:", datos_HP["tablas"])

# Guardamos los datos en un archivo
import json
with open("datos_HP.json", "w", encoding="utf-8") as archivo:
        json.dump(datos_HP, archivo, ensure_ascii=False, indent=4)
   
# Limpiamos los datos obtenidos
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Abre el archivo JSON y carga los datos
with open("datos_HP.json", "r", encoding="utf-8") as archivo:
    datos_HP = json.load(archivo)

# Seleccionamos la tabla que nos interesa y la convertimos en un DataFrame
tabla_HP = datos_HP["tablas"][1]
datos_pelis= pd.DataFrame(tabla_HP)
# Eliminamos las tres columnas de la derecha
datos_pelis = datos_pelis.iloc[:,:-3]

# Convertimos la primera fila en encabezados
datos_pelis.columns = datos_pelis.iloc[0]
datos_pelis = datos_pelis[2:]

# Reiniciamos el índice del DataFrame
datos_pelis.reset_index(drop=True, inplace=True)

# Eliminamos la última fila del dataframe
datos_pelis = datos_pelis.iloc[:-1,:]

# Renombramos las columnas
datos_pelis.rename(columns={
     "Recaudación (en USD)":"Recaudación USA (USD)", "Posición de recaudaciones mundialmente":"Recaudación internacional (USD)"}, inplace=True)

#  Necesitamos que cada columna esté en su formato correcto, es decir, las columnas numéricas (presupuesto y recaudación) deben ser de tipo numérico 
# y la columna de estreno debe ser de tipo fecha.
#Primero hay que convertir la fecha a formato numérico
datos_pelis["Estreno"] = datos_pelis["Estreno"].str.replace(" de ", "-")
mes_a_numero = {
    'enero': '01', 'febrero': '02', 'marzo': '03', 
    'abril': '04', 'mayo': '05', 'junio': '06',
    'julio': '07', 'agosto': '08', 'septiembre': '09',
    'octubre': '10', 'noviembre': '11', 'diciembre': '12'
}
def reemplazar_mes_numero(texto):
    for mes, numero in mes_a_numero.items():
        texto = texto.replace(mes, numero)
    return texto
datos_pelis['Estreno'] = datos_pelis['Estreno'].apply(reemplazar_mes_numero)

datos_pelis["Estreno"] = pd.to_datetime(datos_pelis["Estreno"], format="mixed", dayfirst=True)

#Mantenemos el formato de fecha en el DataFrame
datos_pelis["Estreno"] = datos_pelis["Estreno"].dt.strftime('%d-%m-%Y')
# Reemplazamos los meses por el número y reemplazamos la palabra "millones" por "000 000"
datos_pelis["Presupuesto (en USD)"] = datos_pelis["Presupuesto (en USD)"].str.replace("millones", "000 000")

# Eliminamos los strings [xxx] de las columnas "Película" y "Presupuesto (en USD)"
datos_pelis["Película"] = datos_pelis["Película"].str.replace(r"\[\d+\]", "", regex=True)
datos_pelis["Presupuesto (en USD)"] = datos_pelis["Presupuesto (en USD)"].str.replace(r"\[\d+\]", "",regex=True)

# Aparece un caracter que no se puede leer "\u200b", vamos a eliminarlo en toda la tabla por si apareciera en otros lugares
def reemplazar(df, columnas, viejo, nuevo):
    """
    Reemplaza todas las ocurrencias de un valor viejo por un nuevo valor en las columnas especificadas de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame en el que se realizará la operación.
        columnas (list): Lista de nombres de las columnas donde se realizará el reemplazo.
        viejo (str): El valor que se desea reemplazar.
        nuevo (str): El nuevo valor que reemplazará al valor viejo.

    Returns:
        pd.DataFrame: El DataFrame con los valores reemplazados.
    """
    for columna in columnas:
        df[columna] = df[columna].str.replace(viejo, nuevo)
    return df

columnas = datos_pelis.columns
reemplazar(datos_pelis, columnas, "\u200b", "")

# Convertimos las columnas numéricas a tipo numérico    
def a_numero(df, columnas):
    """
    Convierte las columnas especificadas de un DataFrame a tipo numérico.

    Args:
        df (pd.DataFrame): El DataFrame que contiene las columnas a convertir.
        columnas (list): Lista de nombres de las columnas que se convertirán a tipo numérico.

    Returns:
        pd.DataFrame: El DataFrame con las columnas especificadas convertidas a tipo numérico.
    """
    for columna in columnas:
        df[columna] = pd.to_numeric(df[columna], errors ="coerce")
    return df

# Quitamos los espacios de las columnas "Presupuesto (en USD)", "Recaudación USA (USD)" y "Recaudación internacional (USD)"
reemplazar(datos_pelis, ["Presupuesto (en USD)", "Recaudación USA (USD)", "Recaudación internacional (USD)"], " ", "")

# convertimos las columnas a tipo numérico
a_numero(datos_pelis, ["Presupuesto (en USD)", "Recaudación USA (USD)", "Recaudación internacional (USD)"])

# Añadimos una columna nueva con la recaudación mundial
datos_pelis["Recaudación mundial (USD)"] = datos_pelis["Recaudación USA (USD)"] + datos_pelis["Recaudación internacional (USD)"]

# Mostramos los datos limpios
print(datos_pelis)

# Guardamos los datos limpios en un archivo
datos_pelis.to_csv("datos_pelis.csv", index=False)

# Usamos streamlit para ver los resultados
import streamlit as st
import plotly.express as px
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
datos_pelis = pd.read_csv("datos_pelis.csv")
# Configurar la página
st.title("Análisis de la Saga de Películas de Harry Potter")
st.markdown("Explora aspectos económicos de la saga de películas de Harry Potter.")

# Sección 1: Big Numbers
st.header("Presupuestos y recaudaciones de las películas")
col1, col2, col3 = st.columns(3)


# Mostrar Big Numbers
pelis = datos_pelis["Película"].count()
presupuesto = f"{round(((datos_pelis["Presupuesto (en USD)"].sum())/1000000),0)} M$"
recaudacion  = f"{round(((datos_pelis["Recaudación mundial (USD)"].sum())/1000000),0)} M$"

col1.metric("Número de películas", pelis)
col2.metric("Presupuesto total", presupuesto)
col3.metric("Recaudación mundial total", recaudacion)

# Sección 2: Gráficos 
st.title("Comparación de Presupuesto y Recaudación por Película")

# Crear el gráfico de barras horizontales agrupadas
fig, ax = plt.subplots()

# Posición de las barras
y = np.arange(len(datos_pelis["Película"])) 
height = 0.35  # Ancho de las barras

# Barras de presupuesto y recaudación
bars1 = ax.barh(y - height/2, datos_pelis["Presupuesto (en USD)"], height, label="Presupuesto")
bars2 = ax.barh(y + height/2, datos_pelis["Recaudación mundial (USD)"], height, label="Recaudación")

ax.set_ylabel("Película")
ax.set_xlabel("Monto en millones de dólares")
ax.set_yticks(y)
ax.set_yticklabels(datos_pelis["Película"])
ax.legend()

st.pyplot(fig)

st.title("Comparación de Recaudación en mercados por Película")
# Crear el gráfico de barras horizontales agrupadas
fig, ax = plt.subplots()

# Posición de las barras
y = np.arange(len(datos_pelis["Película"])) 
height = 0.35  # Ancho de las barras

# Barras de presupuesto y recaudación
bars1 = ax.barh(y - height/2, datos_pelis["Recaudación USA (USD)"], height, label="Recaudación USA")
bars2 = ax.barh(y + height/2, datos_pelis["Recaudación internacional (USD)"], height, label="Recaudación internacional")

ax.set_ylabel("Película")
ax.set_xlabel("Monto en millones de dólares")
ax.set_yticks(y)
ax.set_yticklabels(datos_pelis["Película"])
ax.legend()

st.pyplot(fig)
