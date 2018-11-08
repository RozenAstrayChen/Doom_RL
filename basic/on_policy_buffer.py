import numpy as np

class Memory():
    def __init__(self):
        self.s_buffer = []
        self._s_buffer = []
        self.a_buffer = []
        self.r_buffer = []

    def append(self, s, _s, a, r):
        self.s_buffer.append(s)
        self._s_buffer.append(_s)
        self.a_buffer.append(a)
        self.r_buffer.append(r)

    def sample(self):
        '''
        s = np.array(self.s_buffer)
        _s = np.array(self._s_buffer)
        a = np.array(self.a_buffer)
        r = np.array(self.r_buffer)
        '''
        s = np.array(self.s_buffer)
        _s = np.array(self._s_buffer)
        a = np.array(self.a_buffer)
        r = np.array(self.r_buffer)
        return (s, _s, a, r)
    def clean(self):
        self.s_buffer = []
        self._s_buffer = []
        self.a_buffer = []
        self.r_buffer = []