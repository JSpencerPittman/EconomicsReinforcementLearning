import numpy as np

class SquareGrid(object):
    def __init__(self, start=None, n=9):
        if start is None:
            self.grid = np.array([[0] * n] * 9)
            self.n = n
        else:
            if type(start) != np.array:
                self.grid = np.array(start)
            else:
                self.grid = start
            self.n = len(start)

    def update(self, r, c, v):
        if not self.inbounds(r,c):
            return None
        self.grid[r][c] = v

    def val(self, r, c):
        if not self.inbounds(r,c):
            return None
        return self.grid[r][c]
    
    def shape(self):
        return (self.n, self.n)
    
    def ravel(self):
        return np.ravel(self.grid)
    
    def inbounds(self, r, c):
        if r < 0 or r >= self.n:
            return False
        if c < 0 or c >= self.n:
            return False
        return True