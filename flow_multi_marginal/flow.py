import torch 
import numpy as np 
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt 




class cubic_interpolation():

    def create_interpolation(t,xs):     
        
        x_splines = []
        y_splines = []
        x_der_splines = []
        y_der_splines = []


        x_splines.append(interpolate.CubicSpline(t, xs[:, 0]))
        x_der_splines.append(interpolate.CubicSpline(t, xs[:, 0]).derivative(1))

        y_splines.append(interpolate.CubicSpline(t, xs[:, 1]))
        y_der_splines.append(interpolate.CubicSpline(t, xs[:, 1]).derivative(1))

        return x_splines, y_splines, x_der_splines, y_der_splines
    




class cubic_interpolation_across_particles():

    def create_interpolation(t,xs):     
         # Shape (501, num_particles, 2)
        
        """ Notice now that we are creating 10 separate trajectories with this mean, but our model processes a batch of 10x2 particles """
        tp, par, dnt = xs.shape

        x_interpolations= []
        y_interpolations= []

        x_deriv_interpolations= []
        y_deriv_interpolations= []

        for par in range(par): 
            x_coords = xs[:, par, 0]
            y_coords = xs[:, par, 1]
            x_interpolations.append(interpolate.CubicSpline(t, x_coords))
            y_interpolations.append(interpolate.CubicSpline(t, y_coords))
            x_deriv_interpolations.append(interpolate.CubicSpline(t, x_coords).derivative(1))
            y_deriv_interpolations.append(interpolate.CubicSpline(t, y_coords).derivative(1))

        return x_interpolations, y_interpolations, x_deriv_interpolations, y_deriv_interpolations
    



class multi_marginal_flow_batch():      
    def __init__(self, x_idx, sigma_max: float, poly_x, poly_y, poly_der_x, poly_der_y):
        self.sigma_max = 0.5
        self.x_idx = x_idx
        self.poly_x = poly_x
        self.poly_y = poly_y
        self.poly_der_x = poly_der_x
        self.poly_der_y = poly_der_y

    def compute_mean_for_int_batch(self, t_batch):
        """
        Compute mean for a batch of time points
        t_batch: shape (batch_size,)
        Returns: shape (batch_size, 2) for x,y coordinates
        """
        
        batch_size = len(t_batch)
        means = np.zeros((batch_size, len(self.poly_x), 2))
        
        
        
        # for batch_idx, t in enumerate(t_batch):
        #     x_agg, y_agg = [], []
        #     for x in self.x_idx:
        #         x_agg.append(self.poly_x[x](t))
        #         y_agg.append(self.poly_y[x](t))
        #     means[batch_idx] = [np.mean(x_agg), np.mean(y_agg)]
        
        # return means

    def compute_var_batch(self, t_batch, sigma_max=0.1):
        """
        Compute variance for a batch of time points
        t_batch: shape (batch_size,)
        Returns: shape (batch_size,)
        """
        return sigma_max * (np.sin(np.pi * t_batch) ** 2) + 1e-10

    def compute_var_derivative_batch(self, t_batch, sigma_max=0.1):
        """
        Compute variance derivative for a batch of time points
        t_batch: shape (batch_size,)
        Returns: shape (batch_size,)
        """
        return np.pi * sigma_max * np.sin(2 * np.pi * t_batch) 

    def sample_xt_batch(self, t_batch):
        """
        Sample x_t for a batch of time points
        t_batch: shape (batch_size,)
        Returns: shape (batch_size, 2)
        """
        mu = self.compute_mean_for_int_batch(t_batch)  # (batch_size, 2)
        var = self.compute_var_batch(t_batch)  # (batch_size,)
        var_expanded = var[:, np.newaxis]  # (batch_size, 1) for broadcasting
        
        noise = np.random.normal(size=mu.shape)  # (batch_size, 2)
        return mu + var_expanded * noise

    def eval_p_der_batch(self, t_batch):
        """
        Evaluate derivative of p for a batch of time points
        t_batch: shape (batch_size,)
        Returns: shape (batch_size, 2)
        """
  
        batch_size = len(t_batch)
        p_der = np.zeros((batch_size, len(self.poly_der_x), 2))
        
        for batch_idx, t in enumerate(t_batch):
            for i in range(len(self.poly_der_x)):
                
                p_der[batch_idx][i][0] = self.poly_der_x[i](t)
                p_der[batch_idx][i][1] = self.poly_der_y[i](t)
        
        return p_der
    
    def eval_p_batch(self, t_batch):
        """
        Evaluate p for a batch of time points
        t_batch: shape (batch_size,)
        Returns: shape (batch_size, 2)
        """
        batch_size = len(t_batch)
        p = np.zeros((batch_size, len(self.poly_x), 2))
        
        for batch_idx, t in enumerate(t_batch):
            for i in range(len(self.poly_x)):
                
                p[batch_idx][i][0] = self.poly_x[i](t)
                p[batch_idx][i][1] = self.poly_y[i](t)
        
        return p


    def conditional_flow_batch(self, t_samp_batch, x_samp_batch):
        """
        Compute conditional flow for a batch of samples
        t_samp_batch: shape (batch_size,)
        x_samp_batch: shape (batch_size, 2)
        Returns: shape (batch_size, 2)
        """
        sigma_prime = self.compute_var_derivative_batch(t_samp_batch)  # (batch_size,)
        
        sigma = self.compute_var_batch(t_samp_batch)  # (batch_size,)
        

        p_prime = self.eval_p_der_batch(t_samp_batch)  # (batch_size, n_particles, 2)
        
        p = self.eval_p_batch(t_samp_batch) # (batch_size, n_particles, 2)
        
        sigma_expanded = sigma[:, None, None].repeat(15, axis=1).repeat(2, axis=2)

        # Expand sigma_prime similarly
        sigma_prime_expanded = sigma_prime[:, None, None].repeat(15, axis=1).repeat(2, axis=2)
        
        return (sigma_prime_expanded / sigma_expanded) * (x_samp_batch - p) + p_prime
    