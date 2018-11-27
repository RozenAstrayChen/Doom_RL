import numpy as np

def shapping(variable_old, variable_current):
    if len(variable_current) == 0:
        # no need shapping
        return 0
    if len(variable_current) == 1:
        # loss health
        if variable_old[0] > variable_current[0]:
            print('loss health')
            return -10
        # got health package
        elif variable_old[0] < variable_current[0]:
            print('got health package')
            return 10
        else:
            return 0