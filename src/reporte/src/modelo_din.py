import rbdl
import numpy as np
from proyectfunctions import *

# Lectura del modelo del robot a partir de URDF (parsing)
modelo = rbdl.loadModel('../urdf/robot_siad7n.urdf')
# Grados de libertad
ndof = modelo.q_size

# Configuracion articular
q = np.array([-0.74,0.86,0.13,-0.88,0.12,-1.09,-0.62])
# Velocidad articular
dq = np.array([0.8, 0.7, 0.8, 0.6, 0.9, 1.0, 1.0])
# Aceleracion articular
ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5, 1.0])

#freq = 20
#dt = 1.0/freq

#robot = Robot(q, dq, ndof, dt)

# Inicializacion de variables
zeros = np.zeros(ndof)          # Vector de ceros
tau   = np.zeros(ndof)          # Vector de torque
g     = np.zeros(ndof)          # Vector de gravedad
c     = np.zeros(ndof)          # Vector de Coriolis+centrifuga
M     = np.zeros([ndof, ndof])  # Matriz de inercia
e     = np.eye(7)               # Vector identidad

# Torque resultante de la configuracion del robot
rbdl.InverseDynamics(modelo, q, dq, ddq, tau)
#robot.send_command(tau)
#q_r=robot.read_joint_positions()
#dq_r=robot.read_joint_velocities()

#print(robot.M)
#print(dq_r)

# Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga
#          y matriz M usando solamente InverseDynamics

rbdl.InverseDynamics(modelo, q, np.zeros(7), np.zeros(7), g)
rbdl.InverseDynamics(modelo, q, dq, np.zeros(7), c)
c = c - g

print('g') 
print( np.round(g,4))
print('\n')
print('c') 
print(np.round(c,4))
print('\n')

rbdl.CompositeRigidBodyAlgorithm(modelo, q, M, update_kinematics = 'true')

print('M') 
print(np.round(M,3))