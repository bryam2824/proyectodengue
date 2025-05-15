# -- coding: utf-8 --
import telebot
from telebot import types
from telebot.handler_backends import StatesGroup, State
from telebot.storage import StateMemoryStorage
import sqlite3
import csv
import os
from modelo_dengue import model, scaler
import subprocess
import json




# Constantes
DB_PATH = 'datos_personales.db'
CSV_PATH = 'datos_numericos.csv'
TOKEN = '7680882229:AAEFJxRTWozoN18mvFU7nJlrSuBUnYfRxe4'  # <-- Reemplaza esto por tu token real

# Inicialización
state_storage = StateMemoryStorage()
bot = telebot.TeleBot(TOKEN, state_storage=state_storage)

def verificar_conexion():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tablas = cursor.fetchall()
        conn.close()
        if tablas:
            print("✅ Conexión exitosa. Tablas encontradas:", tablas)
        else:
            print("⚠ Conexión establecida, pero no hay tablas en la base de datos.")
    except sqlite3.Error as e:
        print(f"❌ Error en la conexión a SQLite: {str(e)}")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            nombre TEXT,
            cedula TEXT PRIMARY KEY,
            edad INTEGER,
            regiones TEXT,
            peso REAL,
            estatura REAL
        )
    ''')
    conn.commit()
    conn.close()


def guardar_en_sqlite(nombre, cedula, edad, regiones, peso, estatura, chat_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM usuarios WHERE cedula = ?', (cedula,))
    if cursor.fetchone():
        bot.send_message(chat_id, "⚠ Ya estás registrado con esa cédula.")
    else:
        try:
            cursor.execute(
                'INSERT INTO usuarios (nombre, cedula, edad, regiones, peso, estatura) VALUES (?, ?, ?, ?, ?, ?)',
                (nombre, cedula, edad, regiones, peso, estatura)
            )
            conn.commit()
            bot.send_message(chat_id, "✅ ¡Datos guardados correctamente!")
        except sqlite3.Error as e:
            bot.send_message(chat_id, f"❌ Error al guardar datos: {str(e)}")
        finally:
            conn.close()


def verificar_csv():
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', newline='') as f:
            reader = csv.reader(f)
            datos = list(reader)
        if datos:
            print(f"✅ El archivo '{CSV_PATH}' existe y contiene datos:")
            for linea in datos:
                print(linea)
        else:
            print(f"⚠ El archivo '{CSV_PATH}' existe pero está vacío.")
    else:
        print(f"❌ El archivo '{CSV_PATH}' no se ha creado.")

def guardar_datos_numericos(valores):
    archivo_existe = os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a', newline='') as archivo:
        writer = csv.writer(archivo)
        # Si el archivo no existía, escribir encabezados (si aplica)
        if not archivo_existe:
            writer.writerow(["Sintoma_1", "Sintoma_2", "Sintoma_3",""])  # Ajusta según tus datos
        writer.writerow(valores)

guardar_datos_numericos([])



def guardar_datos_numericos(valores):
    with open(CSV_PATH, 'a', newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(valores)

verificar_conexion()
verificar_csv()
init_db()

### Flujo de datos personales
usuarios_datos = {}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add(
        types.KeyboardButton("📋 Ingresar datos personales"),
        types.KeyboardButton("🏥 Diagnostico")
       
    )
    bot.send_message(message.chat.id, "¡Bienvenido!\nPor favor, elige una opción:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == "📋 Ingresar datos personales")
def ask_name(message):
    usuarios_datos[message.chat.id] = {}
    bot.send_message(message.chat.id, "¡Hola! ¿Cuál es tu nombre?")
    bot.register_next_step_handler(message, ask_cedula)

def ask_cedula(message):
    usuarios_datos[message.chat.id]['nombre'] = message.text  # Guarda el nombre correctamente
    bot.send_message(message.chat.id, "🔸 Ingresa tu cédula:")
    bot.register_next_step_handler(message, ask_age)

def ask_age(message):
    usuarios_datos[message.chat.id]['cedula'] = message.text  # Ahora almacena la cédula correctamente
    bot.send_message(message.chat.id, "🔸 Ingresa tu edad:")
    bot.register_next_step_handler(message, ask_age1)
    

def ask_age1(message):
   
    try:
        edad = int(message.text)
        if edad > 0:
            usuarios_datos[message.chat.id]['edad'] = edad  # ✅ Solo almacena la edad
            bot.send_message(message.chat.id, "🔸 Indica a qué región perteneces:")
            ask_regiones(message)
        else:
            raise ValueError
    except ValueError:
        bot.send_message(message.chat.id, "⚠ Ingresa una edad válida.")
        bot.register_next_step_handler(message, ask_age)



def ask_regiones(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    regiones = ["Andina", "Amazonia", "Caribe", "Pacifica", "Los Llanos", "Insular"]
    
    for region in regiones:
        markup.add(types.InlineKeyboardButton(region, callback_data=region))
    
    bot.send_message(message.chat.id, "🔸 Selecciona una región:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data in ["Andina", "Amazonia", "Caribe", "Pacifica", "Los Llanos", "Insular"])
def handle_region_selection(call):
    usuarios_datos[call.message.chat.id]['regiones'] = call.data
    bot.send_message(call.message.chat.id, "🔸 Ingresa tu peso (kg):")
    bot.register_next_step_handler(call.message, ask_peso)


def ask_peso(message):
    try:
        peso = float(message.text)
        if peso > 0:
            usuarios_datos[message.chat.id]['peso'] = peso
            bot.send_message(message.chat.id, "🔸 Ingresa tu estatura (ej: 1.75):")
            bot.register_next_step_handler(message, ask_estatura)
        else:
            raise ValueError
    except ValueError:
        bot.send_message(message.chat.id, "⚠ Ingresa un peso válido.")
        bot.register_next_step_handler(message, ask_peso)

def ask_estatura(message):
    try:
        estatura = float(message.text)
        if estatura > 0:
            usuarios_datos[message.chat.id]['estatura'] = estatura
            info = usuarios_datos[message.chat.id]
            guardar_en_sqlite(
                info['nombre'], info['cedula'], info['edad'],
                info['regiones'], info['peso'], info['estatura'], message.chat.id
            )
        else:
            raise ValueError
    except ValueError:
        bot.send_message(message.chat.id, "⚠ Ingresa una estatura válida.")
        bot.register_next_step_handler(message, ask_estatura)

### Flujo de datos numéricos
datos_numericos = {}
sintomas = ["el dolor en articulaciones y huesos", "el sangrado", "la fiebre"," "]

@bot.message_handler(func=lambda m: m.text == "🏥 Diagnostico")
def inicio_datos_numericos(message):
    datos_numericos[message.chat.id] = {"decimals": [], "integer": None}
    bot.send_message(message.chat.id, "¡Hola! Se te harán preguntas sobre síntomas.")
    bot.send_message(message.chat.id, f"Del 0 a 1, ¿Qué tan fuerte es {sintomas[0]}?")
    escala = "0 a 0.25: Leve\n0.26 a 0.5: Moderado\n0.51 a 0.75: Fuerte\n0.76 a 1: Muy fuerte"
    bot.send_message(message.chat.id, escala)

@bot.message_handler(func=lambda m: m.chat.id in datos_numericos and len(datos_numericos[m.chat.id]["decimals"]) < len(sintomas))
def handle_decimal(message):
    try:
        valor = float(message.text)
        if 0 <= valor <= 1:
            datos_numericos[message.chat.id]["decimals"].append(valor)
            if len(datos_numericos[message.chat.id]["decimals"]) < len(sintomas):
                siguiente = len(datos_numericos[message.chat.id]["decimals"])
                bot.send_message(message.chat.id, f"Del 0 a 1, ¿Qué tan fuerte es {sintomas[siguiente]}?")
                escala = "0 a 0.25: Leve\n0.26 a 0.5: Moderado\n0.51 a 0.75: Fuerte\n0.76 a 1: Muy fuerte"
                bot.send_message(message.chat.id, escala)
            else:
                datos_numericos[message.chat.id]["integer"] = 1  # Asignación automática
                guardar_datos_numericos(datos_numericos[message.chat.id]['decimals'])  # Guarda los datos
                if len(datos_numericos[message.chat.id]["decimals"]) != 4:
                    bot.send_message(message.chat.id, "⚠ Error: Se esperaban 4 datos, pero no se recibieron correctamente.")
                    return
                bot.send_message(message.chat.id, "✅ Datos numéricos registrados. Ejecutando análisis...")
                resultado = ejecutar_modelo(datos_numericos[message.chat.id]['decimals'])
                if resultado is None:
                    bot.send_message(message.chat.id, "⚠ Error en el análisis. Intenta nuevamente más tarde.")
                else:
                    bot.send_message(message.chat.id, f"🏥 Probabilidad de dengue: {resultado:.2f}%")
        else:
            raise ValueError
    except ValueError:
        bot.send_message(message.chat.id, "❌ Debe ser un número entre 0 y 1.")


def verificar_datos():
    if not os.path.exists("datos_usuario.json"):
        print("❌ Error: El archivo 'datos_usuario.json' no existe.")
        return False

    with open("datos_usuario.json", "r") as f:
        datos = json.load(f)

    if not datos or not isinstance(datos, dict) or len(datos) == 0:
        print("⚠ Advertencia: 'datos_usuario.json' está vacío o mal estructurado.")
        return False

    return True

if verificar_datos():
    subprocess.run(["python", "modelo_dengue.py", "datos_usuario.json"])
else:
    print("⏳ Esperando datos válidos antes de ejecutar el modelo...")

def ejecutar_modelo(valores):
    import numpy as np
    try:
        # Asegurar que los valores sean un array 2D con forma (1, 4)
        valores_np = np.array(valores).reshape(1, 4)
        valores_scaled = scaler.transform(valores_np)

        probabilidad = model.predict(valores_scaled)[0][0] * 100  # Ajustar índice si es necesario
        return probabilidad

    except ValueError as e:
        print(f"⚠ Error: Se esperaban 4 valores, pero se recibieron {len(valores)}. {e}")
        return None
    except Exception as e:
        print(f"⚠ Error al ejecutar el modelo: {e}")
        return None



print("🤖 Bot en funcionamiento...")
bot.infinity_polling()
