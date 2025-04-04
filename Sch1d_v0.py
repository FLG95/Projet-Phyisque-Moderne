import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(o, final_density[j,:]) #Crée un graphique pour chaque densite sauvegarde
    return line,

dt=1E-7
dx=0.001
nx=int(1/dx)*2
nt=90000 # En fonction du potentiel il faut modifier ce parametre car sur certaines animations la particule atteins les bords
n_frames=int(nt/1000)+1#nombre d image dans notre animation
s=dt/(dx**2)
v0=-4000
e=5#Valeur du rapport E/V0
E=e*v0
k=math.sqrt(2*abs(E))

x_array = np.linspace(0, (nx - 1) * dx, nx)
V_potential = np.zeros(nx)

#gaussian wave packet (Paquet ondes gaussien)
xc=0.6
sigma=0.05
normalisation=1/(math.sqrt(sigma*math.sqrt(math.pi)))
wp_gauss = normalisation * np.exp(1j * k * x_array - ((x_array - xc) ** 2) / (2 * (sigma ** 2)))
#wave packet Real part 
wp_re=np.zeros(nx)
wp_re[:]=np.real(wp_gauss[:])
#wave packet Imaginary part 
wp_im=np.zeros(nx)
wp_im[:]=np.imag(wp_gauss[:])

density = np.zeros((nt,nx))
density[0,:] = np.absolute(wp_gauss[:]) ** 2

final_density =np.zeros((n_frames,nx))
#Algo devant retourner la densité de probabilité de présence de la particule à différents instants

plot_title = "E/Vo="+str(e)

fig = plt.figure() # initialise la figure principale
line, = plt.plot([], [])
plt.ylim(-1,1)
plt.xlim(0,2)
plt.plot(x_array,V_potential,label="Potentiel")
plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")
#plt.legend() #Permet de faire apparaitre la legende

ani = animation.FuncAnimation(fig,animate,init_func=init, frames=n_frames, blit=False, interval=100, repeat=False)
#file_name = 'paquet_onde_e='+str(e)+'.mp4'
#ani.save(file_name, writer = animation.FFMpegWriter(fps=120, bitrate=5000))
plt.show()
