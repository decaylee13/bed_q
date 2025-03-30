import random

class Patient: 
    def __init__(self, id, severity=None):
        self.id = id
        self.wait_time = 0
        if severity is not None:
            self.severity = severity
        else:
            if random.random() < 0.7:
                self.severity = random.randint(1, 5)
            else:
                self.severity = random.randint(6, 10)
    
    def increase_wait_time(self, time_increment=1):
        self.wait_time += time_increment

    def get_features(self):
        return [self.wait_time, self.severity]