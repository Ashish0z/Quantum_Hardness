import numpy as np

class schedule:
    def __init__(self, T):
        self.T = T
    
    def _at_T(self):
        return 1, 0
    
    def __str__(self) -> str:
        pass

class linear_schedule(schedule):
    def _at_t(self, t):
        x = t/self.T
        A = x
        B = 1 - A
        return A, B
    
    def __str__(self) -> str:
        return "linear"

class quadratic_schedule(schedule):
    def _at_t(self, t):
        x = t/self.T
        A = x**2 * (3 - 2*x)
        B = 1 - A
        return A, B
    
    def __str__(self) -> str:
        return "quadratic"

class cubic_schedule(schedule):
    def _at_t(self, t):
        x = t/self.T
        A = x**3 * (10 + x*(6*x - 15))
        B = 1 - A
        return A, B
    
    def __str__(self) -> str:
        return "cubic"
    
class biquadratic_schedule(schedule):
    def _at_t(self, t):
        x = t/self.T
        A = x**4 * (35 + x*(-84 + x*(70 + x*(-20))))
        B = 1 - A
        return A, B
    
    def __str__(self) -> str:
        return "biquadratic"

class cosine_schedule(schedule):
    def _at_t(self, t):
        x = t/self.T
        A = 0.5*(1 - np.cos(np.pi*(x**2)))
        B = 1 - A
        return A, B
    
    def __str__(self) -> str:
        return "cosine"
