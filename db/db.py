import telebot
import sqlite3
from telebot import types
from telebot.handler_backends import StatesGroup, State
from telebot.storage import StateMemoryStorage

# Configuración
state_storage = StateMemoryStorage()
bot = telebot.TeleBot('7680882229:AAEFJxRTWozoN18mvFU7nJlrSuBUnYfRxe4', state_storage=state_storage)

conn = sqlite3.connect('telegram_bot.db') #db crea la extencion de la base de datos
# Configuración de SQLite
def init_db():
    conn = sqlite3.connect('datos_personales.db')
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


# Guardar datos personales en SQLite
def guardar_datos_personales(cédula, nombres, edad, región, peso, estatura):
    conn = sqlite3.connect('datos_personales.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO usuarios (cédula, nombres, edad, región, peso, estatura)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (cédula, nombres, edad, región, peso, estatura))
    conn.commit()
    conn.close()

def guardar_datos_personales(nombre, cedula, edad, regiones, peso, estatura, chat_id):
    conn = sqlite3.connect('datos_personales.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM usuarios WHERE cedula = ?', (cedula,))
    
    if cursor.fetchone():
        bot.send_message(chat_id, "⚠️ Ya estás registrado con esa cédula.")
    else:
        try:
            cursor.execute(
                'INSERT INTO usuarios (nombre, cedula, edad, regiones, peso, estatura) VALUES (?, ?, ?, ?, ?, ?)',
                (nombre, cedula, edad, regiones, peso, estatura)
            )
            conn.commit()
            bot.send_message(chat_id, "✅ ¡Datos guardados correctamente!")
        except Exception as e:
            bot.send_message(chat_id, f"❌ Error al guardar datos: {str(e)}")
        finally:
            conn.close()