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

#almacenemos datinhos
fqact = open("/tmp/qactual_opg.txt", "w")
fxact = open("/tmp/xactual_opg.txt", "w")
fxdes = open("/tmp/xdeseado_opg.txt", "w")


# Nombres de las articulaciones
jnames = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

# Objeto (mensaje) de tipo JointState
jstate = JointState()
# Valores del mensaje
jstate.header.stamp = rospy.Time.now()
jstate.name = jnames

# =============================================================
# Configuracion articular inicial (en radianes)
q = np.array([0., 0., 0.1, 1, 0.1, -0.5, 0.])
# Velocidad inicial
dq = np.array([0., 0., 0., 0., 0., 0., 0.])
# Aceleracion inicial
ddq = np.array([0., 0., 0., 0., 0., 0., 0])

# Configuracion articular deseada
vdes = np.zeros(3)
ades = np.zeros(3)
# =============================================================

# Posicion resultante de la configuracion articular deseada
xdes = np.array([0.4, 1, 1])
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
valores = 0.1*np.array([1.0, 1.0, 1.0])
Kp = np.diag(valores)
Kd = 2*np.sqrt(Kp)

q_prev = copy(q)
x_prev = np.zeros(3)

# Bucle de ejecucion continua
t = 0.0

b = np.zeros(ndof)          # Para efectos no lineales
M = np.zeros([ndof, ndof])  # Para matriz de inercia

x_prev = fkine(q)[0:3,3]
q_prev = copy(q)
while not rospy.is_shutdown():

    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    print(q)
    # Posicion actual del efector final
    x = fkine(q)[0:3,3]
    
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()
    
    fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
    fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
    fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+' '+str(q[6])+'\n ')


    # ----------------------------
    # Control dinamico 
    # ----------------------------
    
    if np.linalg.norm(x- xdes) < 0.001:
        break

    rbdl.NonlinearEffects(modelo, q, dq, b)    
    rbdl.CompositeRigidBodyAlgorithm(modelo, q, M, update_kinematics = 'true')

    J = jacobian_position(q, dt)
    dJ = J - jacobian_position(q_prev, dt)                
    dJ=dJ/dt
    v = (x - x_prev)/dt

    u = np.dot(M, np.dot( np.linalg.pinv(J), ades - np.dot(dJ, dq) + np.dot(Kd, (vdes - v)) + np.dot(Kp, (xdes - x)) )) + b 

    print('error')
    print(xdes - x)
    q_prev = copy(q)
    x_prev = copy(x)
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