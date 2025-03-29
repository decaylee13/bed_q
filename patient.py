class Patient: 
    def __init__(self):
        self.wait_time = 0
        self.status = None #waiting, in bed, out of bed
    
    def increase_wait_time(self, time_increment=1):
        self.wait_time += time_increment

