import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask_mysqldb import MySQL
from werkzeug.security import check_password_hash
from tensorflow.keras.models import load_model
from datetime import datetime

# Configuración de Flask
app = Flask(__name__)
app.secret_key = 'clave_secreta'

# Configuración de MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Cambia si es necesario
app.config['MYSQL_PASSWORD'] = 'admin123'   # Tu contraseña de MySQL
app.config['MYSQL_DB'] = 'sisol_demo'

mysql = MySQL(app)

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Modelo Keras
model = load_model('modelo_pneumonia.h5', compile=False)
IMG_SIZE = 150

# Clase Doctor para Flask-Login
class Doctor(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@app.route('/')
def home():
    return redirect(url_for('login'))

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, username, password FROM doctores WHERE id = %s", (user_id,))
    doctor = cur.fetchone()
    cur.close()
    if doctor:
        return Doctor(*doctor)
    return None

# Ruta de login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, username, password FROM doctores WHERE username = %s", (username,))
        doctor = cur.fetchone()
        cur.close()

        if doctor and check_password_hash(doctor[2], password_input):
            user = Doctor(*doctor)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Credenciales incorrectas')

    return render_template('login.html')

# Ruta de logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Dashboard protegido
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

# Diagnóstico protegido
@app.route('/diagnostico', methods=['GET', 'POST'])
@login_required
def diagnostico():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            in_memory_file = file.read()
            npimg = np.frombuffer(in_memory_file, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            pred = model.predict(img)[0][0]
            if pred >= 0.5:
                result = f"PNEUMONIA DETECTADA - Prob: {pred:.4f}"
            else:
                result = f"PULMÓN SANO - Prob: {pred:.4f}"

    return render_template('diagnostico.html', result=result)


# Registro de pacientes
@app.route('/pacientes', methods=['GET', 'POST'])
@login_required
def pacientes():
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        nombre = request.form['nombre']
        dni = request.form['dni']
        fecha_nacimiento = request.form['fecha_nacimiento']
        telefono = request.form['telefono']
        direccion = request.form['direccion']

        cur.execute("""
            INSERT INTO pacientes (nombre, dni, fecha_nacimiento, telefono, direccion, doctor_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (nombre, dni, fecha_nacimiento, telefono, direccion, current_user.id))
        mysql.connection.commit()
        flash('Paciente registrado exitosamente')

    cur.execute("SELECT * FROM pacientes WHERE doctor_id = %s", (current_user.id,))
    pacientes = cur.fetchall()
    cur.close()

    return render_template('pacientes.html', pacientes=pacientes)

# Registro de citas
@app.route('/citas', methods=['GET', 'POST'])
@login_required
def citas():
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        paciente_id = request.form['paciente_id']
        fecha = request.form['fecha']
        motivo = request.form['motivo']

        cur.execute("""
            INSERT INTO citas (paciente_id, fecha, motivo)
            VALUES (%s, %s, %s)
        """, (paciente_id, fecha, motivo))
        mysql.connection.commit()
        flash('Cita registrada exitosamente')

    # Obtener pacientes del doctor
    cur.execute("SELECT id, nombre FROM pacientes WHERE doctor_id = %s", (current_user.id,))
    pacientes = cur.fetchall()

    # Listar citas
    cur.execute("""
        SELECT citas.id, pacientes.nombre, citas.fecha, citas.motivo
        FROM citas
        JOIN pacientes ON citas.paciente_id = pacientes.id
        WHERE pacientes.doctor_id = %s
        ORDER BY citas.fecha DESC
    """, (current_user.id,))
    citas = cur.fetchall()

    cur.close()

    return render_template('citas.html', pacientes=pacientes, citas=citas)



if __name__ == '__main__':
    app.run(debug=True)