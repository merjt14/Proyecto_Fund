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

#almacenamos datinhos
fqact = open("/tmp/qactual_pdg.txt", "w")
fqdes = open("/tmp/qdeseado_pdg.txt", "w")
fxact = open("/tmp/xactual_pdg.txt", "w")
fxdes = open("/tmp/xdeseado_pdg.txt", "w")

# Nombres de las articulaciones
jnames = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# Objeto (mensaje) de tipo JointState
jstate = JointState()
# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
# Configuracion articular inicial (en radianes)
q = np.array([0., 0., 0.2, np.pi/2, 0., 0., 0.])
# Velocidad inicial
dq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Configuracion articular deseada
qdes = np.array([0.2, 0.2, 0.3, 1.2, 0.25, 0.2, 0.2])
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = fkine(qdes)[0:3,3]
# Copiar la configuracion articular en el mensaje a ser publicado
jstate.position = q
pub.publish(jstate)

# Modelo RBDL
modelo = rbdl.loadModel('../urdf/robot_siad7n.urdf')
ndof   = modelo.q_size     # Grados de libertad
g      = np.zeros(ndof)    # Espacio para el vecor de gravedad

# Frecuencia del envio (en Hz)
freq = 20
dt = 1.0/freq
rate = rospy.Rate(freq)

# Simulador dinamico del robot
robot = Robot(q, dq, ndof, dt)
# Se definen las ganancias del controlador
valores = 1*np.array([0.1, 1.0, 10, 1.0, 0.01, 0.01, 1e-10])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

u = np.zeros(ndof)   # Espacio para la ley de control
b = np.zeros(ndof)          # Para efectos no lineales
M = np.zeros([ndof, ndof])  # Para matriz de inercia

# Bucle de ejecucion continua
t = 0.0

while not rospy.is_shutdown():

    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    x = fkine(q)[0:3,3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
    fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
    fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+' '+str(q[6])+'\n ')
    fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' '+ str(qdes[2])+' '+ str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+' '+str(qdes[6])+'\n ')

    # ----------------------------
    # Control dinamico
    # ----------------------------

    if np.linalg.norm(q- qdes) < 0.05:
        break

    rbdl.InverseDynamics(modelo, q, np.zeros(ndof), np.zeros(ndof), g)    
    rbdl.CompositeRigidBodyAlgorithm(modelo,q,M)
    rbdl.NonlinearEffects(modelo,q,dq,b)

    u = M.dot(-Kd.dot(dq) + Kp.dot(qdes - q)) + b

    print('error: ')
    print(qdes - q)
    
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
