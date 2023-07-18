import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import datetime
import scipy.special as sp
from datetime import timedelta
from pathlib import Path
from mosgim2.mcoords.mag_coords import geo2mag


def make_matrix(nbig, mbig, theta, phi):
    n_ind = np.arange(0, nbig + 1, 1)
    m_ind = np.arange(-mbig, mbig + 1, 1)
    M, N = np.meshgrid(m_ind, n_ind)
    Y = sp.sph_harm(np.abs(M), N, 0, 0)
    idx = np.isfinite(Y)
    M = M[idx]
    N = N[idx]
    n_coefs = len(M)
    matrix = np.zeros((len(theta), n_coefs))
    for i in range(0, len(theta), 1):
        Ymn = sp.sph_harm(np.abs(M), N, theta[i], phi[i])
        a = np.zeros(len(Ymn))
        a[M < 0] = Ymn[M < 0].imag * np.sqrt(2) * (-1.) ** M[M < 0]
        a[M > 0] = Ymn[M > 0].real * np.sqrt(2) * (-1.) ** M[M > 0]
        a[M == 0] = Ymn[M == 0].real
        matrix[i, :] = a[:]
    return matrix


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot GIMs ')
    parser.add_argument('--in_file',
                        type=Path,
                        help='Path to data, map creation', default='/home/teufel/PycharmProjects/mosgim2/tmp/res1_hdf.npz')
    parser.add_argument('--out_file',
                        type=Path,
                        help='Path to video', default='/home/teufel/PycharmProjects/mosgim2/animation.gif')
    args = parser.parse_args()


    data = np.load(args.in_file, encoding='bytes', allow_pickle=True)

    nbig_layer1 = data['nbig_layer1'] 
    mbig_layer1 = data['mbig_layer1']
    nbig_layer2 = data['nbig_layer2']
    mbig_layer2 = data['mbig_layer2']
    res = data['res']
    nMaps = data['nmaps']  # number of maps
    linear = data['linear']

    # prepare net to estimate TEC on it
    colat = np.arange(2.5, 180, 2.5)
    lon = np.arange(-180, 185, 5.)
    lon_m, colat_m = np.meshgrid(lon, colat)

    nT = nMaps - 1 if linear else nMaps
    step = 86400 / nT


    Z1l = []
    Z2l = []
    Z3l = []

    for k in np.arange(0,nMaps,1): # consecutive tec map number
        print(data['time0'] + timedelta(seconds=step*k))
        mcolat, mt = geo2mag(np.deg2rad(colat_m.flatten()), np.deg2rad(lon_m.flatten()), data['time0'] + timedelta(seconds=step*k)) 


        Atest1 = make_matrix(nbig_layer1, mbig_layer1, mt, mcolat)
        Atest2 = make_matrix(nbig_layer2, mbig_layer2, mt, mcolat)

        Z1 = np.dot(Atest1, res[(0+k)*(len(Atest1[0])+len(Atest2[0])):(0+k)*(len(Atest1[0])+len(Atest2[0]))+len(Atest1[0])]).reshape(len(colat), len(lon))
        Z2 = np.dot(Atest2, res[(0+k)*(len(Atest1[0])+len(Atest2[0]))+len(Atest1[0]):(0+k)*(len(Atest1[0])+len(Atest2[0]))+len(Atest1[0])+len(Atest2[0])]).reshape(len(colat), len(lon))

        Z1l.append(Z1)
        Z2l.append(Z2)
        Z3l.append(Z1+Z2)


    def some_data(i):   # function returns a 2D data array
        return Z1l[i], Z2l[i], Z3l[i] 

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    m1 = np.max(np.array(Z1l))    
    m2 = np.max(np.array(Z2l))    
    m3 = np.max(np.array(Z3l))    
    levels1=np.arange(-0.5,m1,0.5)
    levels2=np.arange(-0.5,m2,0.5)
    levels3=np.arange(-0.5,m3,0.5)
    cont1 = ax1.contourf(lon_m, 90.-colat_m, some_data(0)[0], levels1,  cmap=plt.cm.jet)    # first image on screen
    cont2 = ax2.contourf(lon_m, 90.-colat_m, some_data(0)[1], levels2,  cmap=plt.cm.jet)    # first image on screen
    cont3 = ax3.contourf(lon_m, 90.-colat_m, some_data(0)[2], levels3,  cmap=plt.cm.jet)    # first image on screen
    ax1.set_title('layer1, '+ str(data['time0'] + timedelta(seconds=step*0)))  
    ax2.set_title('layer2, '+ str(data['time0'] + timedelta(seconds=step*0)))  
    ax3.set_title('GIM, '+ str(data['time0'] + timedelta(seconds=step*0)))  
    fig.colorbar(cont1, ax=ax1)
    fig.colorbar(cont2, ax=ax2)
    fig.colorbar(cont3, ax=ax3)
    plt.tight_layout()
    # animation function
    def animate(i):
        global cont1, cont2, cont3
        z1, z2, z3 = some_data(i)
        for c1 , c2 ,c3 in zip(cont1.collections, cont2.collections, cont3.collections):
            c1.remove()  # removes only the contours, leaves the rest intact
            c2.remove()  # removes only the contours, leaves the rest intact
            c3.remove()  # removes only the contours, leaves the rest intact

        cont1 = ax1.contourf(lon_m, 90.-colat_m, z1, levels1,  cmap=plt.cm.jet)
        cont2 = ax2.contourf(lon_m, 90.-colat_m, z2, levels2,  cmap=plt.cm.jet)
        cont3 = ax3.contourf(lon_m, 90.-colat_m, z3, levels3,  cmap=plt.cm.jet)
        ax1.set_title('layer1, '+ str(data['time0'] + timedelta(seconds=step*i)))  
        ax2.set_title('layer2, '+ str(data['time0'] + timedelta(seconds=step*i)))  
        ax3.set_title('GIM, '+ str(data['time0'] + timedelta(seconds=step*i)))  
        return cont1, cont2, cont3

    anim = animation.FuncAnimation(fig, animate, frames=nMaps, repeat=False)
    anim.save(str(args.out_file), writer='imagemagick')

  

