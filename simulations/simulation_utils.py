import time

class ProgressBar():

    def __init__(self, n_total: int, digits: int=5) -> None:
        self.n_total = n_total
        self.t0 = time.time()
        self.digits = digits
        self.n = 0
        self.dt = 0
        self.show()

    def update(self, k=1):
        self.n += 1
        self.dt = time.time() - self.t0
        self.show()
        
    
    def show(self):
        dt_str = time.strftime("%Hh%Mm%Ss", time.gmtime(self.dt))
        print(f"\rProgress: [{'#'*int(self.n/self.n_total*50):50s}] {self.n:{self.digits}d} / {self.n_total:{self.digits}d} {int(self.n/self.n_total*100):5.1f}% {dt_str}", end="")
        
        