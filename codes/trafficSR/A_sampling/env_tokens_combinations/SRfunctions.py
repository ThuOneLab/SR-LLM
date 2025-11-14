import torch
import numpy as np

INF = 1e6
EPSILON = 1e-3
EXP_THRESHOLD = 80
exp_plateau = np.exp(EXP_THRESHOLD)

def add(x, y):
    return torch.add(x, y)


def multiply(x, y):
    return torch.multiply(x, y)


def subtract(x, y):
    return torch.subtract(x, y)


def divide(x, y, epsilon=1e-3):
    zero_mask = abs(y) < epsilon
    sign_mask = torch.sign(y)
    y = torch.where(zero_mask, sign_mask * y, y)
    return torch.divide(x, y)


def exp(x, sup=10):
    return torch.exp(torch.min(x, torch.tensor(sup)))


def log(x, inf=1e-6):
    return torch.log(torch.max(x, torch.tensor(inf)))


def sin(x):
    return torch.sin(x)


def cos(x):
    return torch.cos(x)


def tan(x):
    return torch.tan(x)


def sinh(x):
    return torch.sinh(x)


def cosh(x):
    return torch.cosh(x)


def tanh(x):
    return torch.tanh(x)


def arctan(x):
    return torch.arctan(x)


def inverse(x, epsilon=1e-3):
    zero_mask = abs(x) < epsilon
    sign_mask = torch.sign(x)
    x = torch.where(zero_mask, sign_mask * x, x)
    return torch.divide(1., x)


def sqrt(x, epsilon=1e-6):
    return torch.where(x > epsilon, torch.sqrt(x), 0.)


def n2(x):
    npow = 2
    sign = 1 if int(npow) % 2 == 0 else torch.sign(x)  # Takes the sign of x if power is odd
    # Caps square to avoid overflow, returns infinity
    limit = torch.pow(torch.tensor(INF), 1 / npow)
    return torch.where(torch.abs(x) < limit, torch.pow(x, npow), sign * INF)


def n3(x):
    npow = 3
    sign = 1 if int(npow) % 2 == 0 else torch.sign(x)  # Takes the sign of x if power is odd
    # Caps square to avoid overflow, returns infinity
    limit = torch.pow(torch.tensor(INF), 1 / npow)
    return torch.where(torch.abs(x) < limit, torch.pow(x, npow), sign * INF)


def n4(x):
    npow = 4
    sign = 1 if int(npow) % 2 == 0 else torch.sign(x)  # Takes the sign of x if power is odd
    # Caps square to avoid overflow, returns infinity
    limit = torch.pow(torch.tensor(INF), 1 / npow)
    return torch.where(torch.abs(x) < limit, torch.pow(x, npow), sign * INF)


def pow(x0, x1):
    # Handles power function, caps at positive/negative infinity to avoid overflow
    if not torch.is_tensor(x0):
        x0 = torch.ones_like(x1) * x0
    # Handle negative bases with non-integer exponents
    result_is_nan = torch.isnan(torch.pow(x0, x1))
    x0 = torch.where(result_is_nan, torch.abs(x0), x0)

    y = torch.pow(x0, x1)
    # Handle overflow
    y = torch.where(torch.abs(y) < INF, y, torch.sign(y) * INF)
    # Handle underflow
    y = torch.where(torch.abs(y) > EPSILON, y, 0.)
    return y


def protected_div(x,y):
    return torch.where(torch.abs(y) > EPSILON, torch.divide(x, y), 1.)
def protected_inverse(x1):
    # with np.errstate(divide='ignore', invalid='ignore'):
    return torch.where(torch.abs(x1) > EPSILON, 1. / x1, 0.)


n2_plateau = np.square(INF)
def protected_n2(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) <= INF, torch.square(x1), n2_plateau)
n3_plateau = np.power(INF, 3)
def protected_n3(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) <= INF, torch.pow(x1, 3), torch.sign(x1)*n3_plateau)
n4_plateau = np.power(INF, 4)
def protected_n4(x1):
    # with np.errstate(over='ignore'):
    return torch.where(torch.abs(x1) <= INF, torch.pow(x1, 4), n4_plateau)
def protected_sqrt(x1):
    return torch.sqrt(torch.abs(x1))

exp_plateau = np.exp(EXP_THRESHOLD)
def protected_exp(x1):
    #with np.errstate(over='ignore'):
    return torch.where(x1 <= EXP_THRESHOLD, torch.exp(x1), exp_plateau)
log_plateau = np.log(np.abs(EPSILON))
def protected_log(x1):
    #with np.errstate(divide='ignore', invalid='ignore'):
    return torch.where(torch.abs(x1) >= EPSILON, torch.log(torch.abs(x1)), log_plateau)

def protected_pow(x0, x1):
    if not torch.is_tensor(x0):
       x0 = torch.ones_like(x1)*x0
    y = torch.pow(x0, x1)
    y = torch.where(y > INF, INF, y)
    return y

def protected_arcsin (x1):
    return torch.where(torch.abs(x1) < (1.-EPSILON), torch.arcsin(x1), torch.sign(x1)*INF)
def protected_arccos (x1):
    return torch.where(torch.abs(x1) < (1.-EPSILON), torch.arccos(x1), torch.sign(x1)*INF)