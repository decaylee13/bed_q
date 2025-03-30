import random
class Patient: 
    def __init__(self, id):
        self.id = id
        self.wait_time = 0
        if random.random() < 0.7:
            self.severity = random.randint(1, 5)
        else:
            self.severity = random.randint(6, 10)
    
    def increase_wait_time(self, time_increment=1):
        self.wait_time += time_increment

    def get_features(self):
        # Later on should include patient feature vector so the model can consider who is waiting for a bed and strategize with that and pure wait time
        return [self.wait_time, self.severity]