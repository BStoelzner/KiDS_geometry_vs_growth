import os
import numpy as np
from montepython import io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as const
import scan_util_16 as util

class eBOSS_DR16_Lya_combined(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
        # needed arguments in order to get derived parameters
        # should be made into an if loop
        # self.need_cosmo_arguments(data, {'output': 'mPk'})
        # self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': 20.})
        # self.need_cosmo_arguments(data, {'z_max_pk': 3.})
        
        # define array for values of z and data points
        self.z = np.array([self.zeff], 'float64')
        scan_locations = self.data_directory + '/' + self.cf_scan

        # number of data points
        self.num_points = np.shape(self.z)[0]

        #Make our interpolators
        self.chi2_interpolators = util.chi2_interpolators(scan_locations,self.transverse_fid,self.parallel_fid)

        # end of initialization

    # compute log likelihood
    def loglkl(self, cosmo1, cosmo2, data):

        chi2 = 0.

        for i in range(self.num_points):

            H  = cosmo2.Hubble(self.z[i]) * const.c / 1000.
            da = cosmo2.angular_distance(self.z[i])
            dm = da * (1 + self.z[i])
            rd = cosmo2.rs_drag() * self.rd_rescale

            transverse = dm / rd
            parallel = (const.c / 1000.) / (H * rd)
            chi2 += self.chi2_interpolators.get_Dchi2_from_distances(transverse,parallel)
        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl
