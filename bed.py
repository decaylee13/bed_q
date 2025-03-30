import time 
import random

class Bed: 
    def __init__(self, bed_id, efficiency):
        self.bed_id = bed_id
        self.efficiency = efficiency
        self.time_occupied = 0 # how long has the patient been in bed? this changes; reset for new patients
        self.occupancy_delta = 0 # how long does the patient need to be in bed? this does not change; reset for new patients
        self.current_patient = None
    
    def time_occupied_increase(self, time_increment = 1):
        self.time_occupied += time_increment

    def assign_patient(self, patient):
        """Assign a patient to this bed."""
        if self.current_patient:
            raise ValueError("Attempting to assign patient to an unavailable bed")
    
        self.current_patient = patient
        self.occupancy_delta = 10 * patient.severity / self.efficiency
        
    def discharge_patient(self):
        """Discharge the current patient and make bed available."""
        self.reset()

    def is_occupied(self):

        return self.current_patient is not None

    def get_features(self):

        # later on may want to include state output of bed feature vector & occupancy delta to help the model strategize more than learning these values

        occupancy = self.is_occupied()

        return [occupancy, self.time_occupied, self.efficiency, self.occupancy_delta] # also time_occupied, self.occupancy_delta, later
    
    def reset(self):
        """Reset the bed to initial state."""
        self.current_patient = None
        self.time_occupied = 0
        self.occupancy_delta = 5


