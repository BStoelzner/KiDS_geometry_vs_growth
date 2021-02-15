import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
import math 
from scipy import special
from scipy.linalg import cholesky, solve_triangular

class CMB_features_2018(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)
	
	#ZP: uncomment if run on its own
	#self.need_cosmo1_arguments(data, {'output': 'mPk'})
        #self.need_cosmo1_arguments(data, {'P_k_max_h/Mpc': '1.'})
        #self.need_cosmo1_arguments(data, {'z_max_pk': '1.'})
	#self.need_cosmo2_arguments(data, {'output': 'mPk'})
        #self.need_cosmo2_arguments(data, {'P_k_max_h/Mpc': '1.'})
        #self.need_cosmo2_arguments(data, {'z_max_pk': '1.'})

        # define array for values of z and data points
        self.data = np.array([], 'float64')
        self.data_type = np.array([], 'int')

        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for line in filein:
               if line.strip() and line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()
                    ## insert into array if this id is not manually excluded
                    self.data = np.append(self.data, float(this_line[1]))
                    self.data_type = np.append(self.data_type, int(this_line[2]))
	#self.data = [3.045,  0.9649, 1.0409]
	#self.data_type = [1, 2, 3]
        # read covariance matrix
        self.covmat = np.loadtxt(os.path.join(self.data_directory, self.cov_file))
        self.cholesky_transform = cholesky(self.covmat, lower=True)

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo1, cosmo2,  data):
        #ZP: Being the position of the acoustic peak 
        # a strong goemetry phenomenon, calculations 
        #are done using cosmo2 as it is the one responsible to 
        #parametrize goemetry. 

        #However, both cosmologies are called for matters 
        #of consistency at the sampler 

        # for each point, compute angular distance da, radial distance dr,
        # volume distance dv, sound horizon at baryon drag rs_d,
        # theoretical prediction and chi2 contribution
        prediction = np.array([], dtype='float64')
        for counter, item in enumerate(self.data):

            if self.data_type[counter] == 1:
                if 'ln10^{10}A_s_1' in data.mcmc_parameters:
                    theo = data.mcmc_parameters['ln10^{10}A_s_1']['current']
                else:
                    theo = log(1.e10*cosmo1.get_A_s)

            elif self.data_type[counter] == 2:
                theo = data.mcmc_parameters['n_s_1']['current']

            elif self.data_type[counter] == 3:
                theo = cosmo2.theta_star_100()
		
            else:
                raise io_mp.LikelihoodError(
                    "In likelihood %s. " % self.name +
                    "BAO data type %s " % self.type[counter] +
                    "in %d-th line not understood" % counter)
	     
            prediction = np.append(prediction, float(theo))

	#print prediction
        vec = self.data - prediction
        if np.isinf(vec).any() or np.isnan(vec).any():
            chi2 = 2e12
        else:
            # don't invert that matrix...
            # use the Cholesky decomposition instead:
            yt = solve_triangular(self.cholesky_transform, vec, lower=True)
            chi2 = yt.dot(yt)
        # return ln(L)
        lkl = -chi2/2.

        return lkl
