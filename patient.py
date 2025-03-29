class Patient: 
    def __init__(self, id):
        self.wait_time = 0
        self.status = None #waiting, in bed, out of bed
        self.id = id
    
    def increase_wait_time(self, time_increment=1):
        self.wait_time += time_increment

    def get_features(self):

        # Later on should include patient feature vector so the model can consider who is waiting for a bed and strategize with that and pure wait time
        
        return [self.wait_time]