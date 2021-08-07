#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from markers import *
from proyectfunctions import *
from roslib import packages

import rbdl


rospy.init_node("control_pdg")
pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
bmarker_actual  = BallMarker(color['RED'])
bmarker_deseado = BallMarker(color['GREEN'])

# Nombres de las articulaciones
jnames = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# Objeto (mensaje) de tipo JointState
jstate = JointState()

# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
# Configuracion articular inicial (en radianes)
q = np.array([-0.74,0.86,0.13,-0.88,0.12,-1.09,-0.62])
# Velocidad inicial
dq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Aceleracion inicial
ddq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Configuracion articular deseada
qdes = np.array([1.0, -1.0, 0.2, 1.3, 0.15, 1.0, -0.88])
# Velocidad articular deseada
dqdes = np.array([0., 0., 0., 0., 0., 0., 0.])
# Aceleracion articular deseada
ddqdes = np.array([0., 0., 0., 0., 0., 0., 0.])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = fkine(qdes)[0:3,3]
# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo RBDL
modelo = rbdl.loadModel('../rsiad7n/robot_siad7n/urdf/robot_siad7n.urdf')
ndof   = modelo.q_size     # Grados de libertad
zeros = np.zeros(ndof)     # Vector de ceros
c_g = np.zeros(ndof)       # Vector de coriolis y gravedad
M = np.zeros([ndof, ndof]) #Matriz de inercia

# Frecuencia del envio (en Hz)
freq = 20
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)

# Bucle de ejecucion continua
t = 0.0

# Se definen las ganancias del controlador
valores = 0.1*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0 , 1.0])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

while not rospy.is_shutdown():

    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    x = fkine(q)[0:3,3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    # ----------------------------
    # Control dinamico 
    # ----------------------------

    rbdl.NonlinearEffects(modelo, q, dq, c_g)    
    rbdl.CompositeRigidBodyAlgorithm(modelo, q, M, update_kinematics = 'true')
    
    u = np.dot(M, ddqdes + np.dot(Kd, (dqdes-dq)) + np.dot(Kp, (qdes-q))) + c_g

    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    jstate.position = q
    pub.publish(jstate)
    bmarker_deseado.xyz(xdes)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()
