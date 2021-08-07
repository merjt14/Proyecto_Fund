import numpy as np
from copy import copy
import rbdl

pi = np.pi
def dh(d, theta, a, alpha):
    """
    Se calcula la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine(q):
    """
    Se calcula la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6, q7] """    
    cos=np.cos; sin=np.sin; pi=np.pi
    # Matrices DH empleando la funcion dh con los parametros DH encontrados
    T1 = dh(       0.4145,     q[0], 0, pi/2)
    T2 = dh(          0.0,  pi+q[1], 0, pi/2)
    T3 = dh(  0.5445+q[2],       pi, 0, pi/2)
    T4 = dh(        0.265,pi/2-q[3], 0, pi/2)
    T5 = dh(   0.488+q[4],    -pi/2, 0, pi/2)
    T6 = dh(-0.1605+0.016,  pi-q[5], 0, pi/2)
    T7 = dh(      0.16219,     q[6], 0,    0)
    
    # Efector final con respecto a la base
    T02 = np.dot( T1,T2)
    T03 = np.dot(T02,T3)
    T04 = np.dot(T03,T4)
    T05 = np.dot(T04,T5)
    T06 = np.dot(T05,T6)
    T07 = np.dot(T06,T7)
    T = T07			
    return T

def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x7 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6, q7]
    """
    # Crear una matriz 3x7
    J = np.zeros((3,7))
    # Transformacion homogenea inicial (usando q)
    Ti = fkine(q)

    # Iteracion para la derivada de cada columna
    for i in xrange(7):
        # Se copia la configuracion articular inicial
        dq = copy(q)
        # Se incrementa la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta    
        # Transformacion homogenea luego del incremento (q+delta)
        Tf = fkine(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[:,i] = (Tf[0:3,3]-Ti[0:3,3])/delta
    return J


def ikine(xdes, q0, error):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la 
    configuracion articular inicial de q0 empleando el metodo de newton.     
    """
    epsilon  = 0.001
    max_iter = 100
    delta    = 0.01

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian_position(q, delta)
        x = fkine(q)
        e = xdes - x[0:3,3]
        
        error.write(str(i)+' '+str(np.round(np.linalg.norm(e),2)) +'\n')

        if np.linalg.norm(e)<epsilon:     
            break
        
        qx = q + np.linalg.pinv(J).dot(e)

        # Limites articulares
        if (qx[0]>1.57):
            q[0]=1.57
        elif (qx[0]<-1.57):
            q[0]=-1.57
        else:
            q[0]=qx[0]

        if (qx[1]>1.57):
            q[1]=1.57
        elif (qx[1]<-1.57):
            q[1]=-1.57
        else:
            q[1]=qx[1]                   

        if (qx[2]>0.32):
            q[2]=0.32
        elif (qx[2]<0):
            q[2]=0
        else:
            q[2]=qx[2]

        if (qx[3]>1.40):
            q[3]=1.40
        elif (qx[3]<-4.54):
            q[3]=-4.54                
        else:
            q[3]=qx[3]

        if (qx[4]>0.25):
            q[4]=0.25
        elif (qx[4]<0):
            q[4]=0
        else:
            q[4]=qx[4]

        if (qx[5]>1.57):
            q[5]=1.57
        elif (qx[5]<-1.57):
            q[5]=-1.57                   
        else:
            q[5]=qx[5]

        if (qx[6]>3.14):
            q[6]=3.14
        elif (qx[6]<-3.14):
            q[6]=-3.14  
        else:
            q[6]=qx[6]                           

    return q

# ESTA FUNCION NO ESTA TOTALMENTE BIEN
def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x7 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6,q7]
    """
    J = np.zeros((7,7))
    # Transformacion homogenea inicial (usando q)
    Ti = fkine(q)
    Qi = rot2quat(Ti[0:3,0:3])

    # Iteracion para la derivada de cada columna
    for i in xrange(7):
        # Se copia la configuracion articular inicial
        dq = copy(q)
        # Se incrementa la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta    
        # Transformacion homogenea luego del incremento (q+delta)
        Tf = fkine(dq)
        Qf = rot2quat(Tf[0:3,0:3])
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0:3,i] = (Tf[0:3,3]-Ti[0:3,3])/delta
        J[3:7,i] = (Qf-Qi)/delta
    return J

def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]

    quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
    if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
    if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
    if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

    return np.array(quat)

def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)

def skew(w):
    '''
    Devuelve la velocidad angular de la matriz antisimetrica de entrada.
    '''
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt  
        self.robot = rbdl.loadModel('../urdf/robot_siad7n.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq

