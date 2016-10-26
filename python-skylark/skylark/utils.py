import skylark.lib as lib



def get_direction(dir):
    if dir == 0 or dir == "rows":
        return 0
    elif dir == 1 or dir == "columns":
        return 1
    else:
        raise ValueError("Direction must be either rows/columns or 0/1")

def adapt_and_check(X):
    X = lib.adapt(X)
    if (X.ptr() == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed")
        
    return X