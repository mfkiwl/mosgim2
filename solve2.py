import numpy as np
from pathlib import Path
from datetime import timedelta
from mosgim2.mosgim.mosgim2 import solve_all

# configuration
nbig_layer1 = 15  # max order of spherical harmonic expansion
mbig_layer1 = 15  # max degree of spherical harmonic expansion (0 <= mbig <= nbig)
nbig_layer2 = 10  # max order of spherical harmonic expansion
mbig_layer2 = 10  # max degree of spherical harmonic expansion (0 <= mbig <= nbig)
tint = 24 # number of time intervals per day 
sigma0 = 0.02  # TECU - measurement noise at zenith (2 mm for phase measurements)
sigma_v = 0.005  # TECU - allowed variability for each coef between two consecutive maps (0.03 TECu by Shaer for dt=2h and 149 coeffs)
linear = True # assumes linear interpolation between time nodes
lcp = True # impose positivity constrains for TEC in each layer by solving LCP 
#################################


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Solve raw TECs to ')
    parser.add_argument('--in_file',
                        type=Path,
                        help='Path to data, after prepare script')
    parser.add_argument('--out_file',
                        type=Path,
                        help='Path to data, after prepare script')
    parser.add_argument('--nworkers',
                        type=int,
                        help='number of workers', default=1)
    args = parser.parse_args()

    inputfile = args.in_file
    outputfile = args.out_file

    data = np.load(inputfile, encoding='bytes', allow_pickle=True)
    ndays = np.ceil((np.max(data['time']) - np.min(data['time'])) / 86400).astype('int') # number of days in input file

    nT_add = 1 if linear else 0

    if not (ndays == 1 or ndays == 3):
        print('procedure only works with 1 or 3 consecutive days data')
        exit(1)


    res, disp_scale, Ninv = solve_all(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, tint, sigma0, sigma_v, data, gigs=2, lcp=lcp, nworkers=args.nworkers, linear=linear)


    if ndays == 1:
        np.savez(outputfile, time0=data['day'], linear=linear, nmaps= tint + nT_add, nbig_layer1=nbig_layer1, mbig_layer1=mbig_layer1, nbig_layer2=nbig_layer2, mbig_layer2=mbig_layer2, res=res, disp_scale=disp_scale)
    if ndays == 3:
        np.savez(outputfile, time0=data['day']+timedelta(days=1), linear=linear, nmaps= tint + nT_add, nbig_layer1=nbig_layer1, mbig_layer1=mbig_layer1, nbig_layer2=nbig_layer2, mbig_layer2=mbig_layer2, 
                 res=res[n_coefs * (tint):n_coefs * (2 * tint + nT_add)], disp_scale=disp_scale)

