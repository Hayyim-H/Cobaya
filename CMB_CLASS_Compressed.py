import numpy as np
from cobaya.likelihood import Likelihood

class CMB_CLASS_Compressed(Likelihood):

    def initialize(self):
        # Datos observados
        self.data_vector = np.array([
            0.01041,
            0.02223,
            0.14208
        ])

        # Matriz de covarianza y su inversa
        self.cov = np.array([[0.006621e-9, 0.12444e-9, -1.1929e-9], 
                             [0.12444e-9, 21.344e-9, -94.001e-9], 
                             [-1.1929e-9, -94.001e-9, 1488.4e-9]])
        self.cov_inv = np.linalg.inv(self.cov)

    def get_requirements(self):
        return {'cosmomc_theta': None, 'omega_b': None, 'ombch2': None}

    def logp(self, **params_values):

        # Obtener otros parámetros necesarios
        cosmomc_theta_t = self.provider.get_param("cosmomc_theta")
        ombh2_t = self.provider.get_param("omega_b")
        ombch2_t = self.provider.get_param("ombch2")

        # Vector de predicciones del modelo
        model_vector = np.array([
            cosmomc_theta_t, ombh2_t, ombch2_t
        ])

        # Calcular chi^2
        delta = model_vector - self.data_vector
        chi2 = delta @ self.cov_inv @ delta

        return -0.5 * chi2

