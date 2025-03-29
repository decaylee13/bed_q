import time 

class Bed: 
    def __init__(self, bed_id):
        self.bed_id = bed_id
        self.is_occupied = False
        self.time_occupied = 0
        self.occupancy_delta = 0
        self.current_patient = None
    
    def time_occupied_increase(self, time_increment = 1):
        self.time_occupied += time_increment

    #Update later
    def calc_occupancy(self, patient): 
        return 30

    def assign_patient(self, patient_id):
        """Assign a patient to this bed."""
        if self.current_patient:
            raise ValueError("Attempting to assign patient to an unavailable bed")
    
        self.current_patient = patient_id
        
    def discharge_patient(self):
        """Discharge the current patient and make bed available."""
        self.is_occupied = False
        self.current_patient = None

    def get_features(self):
        return [self.is_occupied]
    
    def reset(self):
        """Reset the bed to initial state."""
        self.available = True
        self.current_patient = None
        self.expected_discharge_time = None


