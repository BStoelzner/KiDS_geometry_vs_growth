import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts
import math
from scipy import special
from scipy.linalg import cholesky, solve_triangular

class bao_fs_boss_dr12(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # needed arguments in order to get sigma_8(z) up to z=1 with correct precision
        if 'K1K_CorrelationFunctions_2cosmos_geo_vs_growth' in data.experiments:
            print('Conflict!- using kids P_K')
        else:
            self.need_cosmo1_arguments(data, {'output': 'mPk'})
            self.need_cosmo1_arguments(data, {'P_k_max_h/Mpc': self.k_max})
            self.need_cosmo2_arguments(data, {'output': 'mPk'})
            self.need_cosmo2_arguments(data, {'P_k_max_h/Mpc': self.k_max})
            self.need_cosmo1_arguments(data, {'z_max_pk': self.k_max})
            self.need_cosmo2_arguments(data, {'z_max_pk': self.k_max})

        # are there conflicting experiments?
        if 'bao_boss_aniso' in data.experiments:
            raise io_mp.LikelihoodError(
                'conflicting bao_boss_aniso measurments')

        # define arrays for values of z and data points
        self.z = np.array([], 'float64')
        self.DM_rdfid_by_rd_in_Mpc = np.array([], 'float64')
        self.H_rd_by_rdfid_in_km_per_s_per_Mpc = np.array([], 'float64')
        self.fsig8 = np.array([], 'float64')

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and D_M * (r_s / r_s_fid)^-1 in Mpc
                    if this_line[1] == 'dM(rsfid/rs)':
                        self.z = np.append(self.z, float(this_line[0]))
                        self.DM_rdfid_by_rd_in_Mpc = np.append(
                            self.DM_rdfid_by_rd_in_Mpc, float(this_line[2]))
                    # load H(z) * (r_s / r_s_fid) in km s^-1 Mpc^-1
                    elif this_line[1] == 'Hz(rs/rsfid)':
                        self.H_rd_by_rdfid_in_km_per_s_per_Mpc = np.append(
                            self.H_rd_by_rdfid_in_km_per_s_per_Mpc, float(this_line[2]))
                    # load f * sigma8
                    elif this_line[1] == 'fsig8':
                        self.fsig8 = np.append(self.fsig8, float(this_line[2]))

        # read covariance matrix
        self.covmat = np.loadtxt(os.path.join(self.data_directory, self.cov_file))
        self.cholesky_transform = cholesky(self.covmat, lower=True)


        # number of bins
        self.num_bins = np.shape(self.z)[0]

        # number of data points
        self.num_points = np.shape(self.covmat)[0]

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo1, cosmo2, data):

        # define array for  values of D_M_diff = D_M^th - D_M^obs and H_diff = H^th - H^obs,
        # ordered by redshift bin (z=[0.38, 0.51, 0.61]) as following:
        # data_array = [DM_diff(z=0.38), H_diff(z=0.38), DM_diff(z=0.51), .., .., ..]
        vec = np.array([], 'float64')

        # for each point, compute comoving angular diameter distance D_M = (1 + z) * D_A,
        # sound horizon at baryon drag rs_d, theoretical prediction
        for i in range(self.num_bins):
            DM_at_z = cosmo2.angular_distance(self.z[i]) * (1. + self.z[i])
            H_at_z = cosmo2.Hubble(self.z[i]) * conts.c / 1000.0
            rd = cosmo2.rs_drag() * self.rs_rescale
            theo_fsig8 = cosmo1.scale_independent_growth_factor_f(self.z[i])*cosmo1.sigma(8./cosmo1.h(),self.z[i])
            #print('f(',self.z[i],') =',cosmo.scale_independent_growth_factor_f(self.z[i]))
            #print('sigma8(',self.z[i],') =',cosmo.sigma(8./cosmo.h(),self.z[i]))
            #print('f*sig8 =',theo_fsig8)

            theo_DM_rdfid_by_rd_in_Mpc = DM_at_z / rd * self.rd_fid_in_Mpc
            theo_H_rd_by_rdfid = H_at_z * rd / self.rd_fid_in_Mpc

            # calculate difference between the sampled point and observations
            DM_diff = theo_DM_rdfid_by_rd_in_Mpc - self.DM_rdfid_by_rd_in_Mpc[i]
            H_diff = theo_H_rd_by_rdfid - self.H_rd_by_rdfid_in_km_per_s_per_Mpc[i]
            fsig8_diff = theo_fsig8 - self.fsig8[i]

            # save to data array
            vec = np.append(vec, DM_diff)
            vec = np.append(vec, H_diff)
            vec = np.append(vec, fsig8_diff)


        if np.isinf(vec).any() or np.isnan(vec).any():
            chi2 = 2e12
        else:
            # don't invert that matrix...
            # use the Cholesky decomposition instead:
            yt = solve_triangular(self.cholesky_transform, vec, lower=True)
            chi2 = yt.dot(yt)

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl
