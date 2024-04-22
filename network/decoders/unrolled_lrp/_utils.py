
def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())

def minmax_dims(x, minmax):
    y = x.clone()
    dims = x.dim()
    for i in range(1, dims):
        y = getattr(y, minmax)(dim=i, keepdim=True)[0]
    return y
