import copy


def check_outside(x, y, width, height):
    if x < 0 or x >= width or y < 0 or y >= height:
        return True
    else:
        return False

def clip_rect(x, y, width, height):
    x_new = max(0, min(x, width-1))
    y_new = max(0, min(y, height-1))
    return x_new, y_new

def new_array(dims, val):
    
    assert(type(dims) is int or type(dims) is tuple or type(dims) is list)

    if type(dims) is int:
        return [val for i in range(dims)]
    elif len(dims) == 1:
        return [val for i in range(dims[0])]
    else:
        return [new_array(dims[1:], val) for i in range(dims[0]) ]