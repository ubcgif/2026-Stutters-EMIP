from simpeg.simulation import BaseSimulation
from simpeg import survey
import numpy as np
import scipy.sparse as sp


# TODO: deprecate this later
class SEInvProblem(BaseSimulation):

    def __init__(self, survey=None, **kwargs):
        BaseSimulation.__init__(self, survey, **kwargs)

    def fields(self, m):
        return self.forward_model(m)

    def dpred(self, m, f=None):
        if f is None:
            f = self.fields(m)
        return f

    def forward_model(self, m):
        t = np.maximum(self.survey.locations, 1e-12)
        
        eta, tau, c = m

        return eta * c / t * ((t / tau)**c) * np.exp(-(t / tau)**c)

    def getJ(self, m):
        t = self.survey.locations
        eta, tau, c = m

        exp_term = np.exp(-(t/tau)**c)
        pow_term = (t/tau)**c

        # d/d eta
        d_eta = c/t * pow_term * exp_term

        # d/d tau
        d_tau = (
            -c**2 * eta * pow_term / (t * tau)
            * exp_term * (-(pow_term) + 1.)
        )

        # d/d c
        d_c = (
            -eta/t * pow_term * exp_term *
            (c * pow_term * np.log(t/tau) - c*np.log(t/tau) - 1.)
        )

        return np.vstack([d_eta, d_tau, d_c]).T  # shape (ndata, 3)
    
    def Jvec(self, m, v, f=None):
        J = self.getJ(m)
        return J @ v

    def Jtvec(self, m, v, f=None):
        J = self.getJ(m)
        return J.T @ v
    

class SESurvey(survey.BaseSurvey):
    def __init__(self, times):
        super().__init__(source_list=[])
        self.locations = times

    @property
    def nD(self):
        return len(self.locations)