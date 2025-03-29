from bed import Bed
from patient import Patient
class HospitalBedEnv(): 
    def __init__(self, config):
        self.config = config 

        self.curr_time = 0 
        self.curr_episode = 0
        self.episode_steps = 0 
        self.max_steps = config.get('max_episode_steps', 1000)

        self.patients = {} #check
        self.beds = self._initialize_beds(config['beds_config']) #check
        self.patient_queue = []
        self.current_patient = None
        
        self.stats = {
            'wait_times': [],
            'specialty_matches': [],
            'bed_utilization': []
        }

    def reset(self): 
        self.curr_time = 0 
        self.curr_episode = 0
        self.episode_steps = 0 

        for beds in self.beds: #check
            beds.reset() 

        self.patients = {}
        self.patient_queue = []
        self.patient_queue.append(Patient(self.curr_time)) #add 1 patient 

        return self.get_state()#check
    
    def step(self, action):
        """
        Process an action and return the next state, reward, and done flag.
        """
        reward = 0
        
        if action == 0:  
            reward = self._handle_wait_action() #check
        else: 
            bed_idx = action - 1 #bed id from 0 to N
            if bed_idx < len(self.beds) and not self.beds[bed_idx].is_occupied():
                reward = self.handle_assignment_action(bed_idx) #check
            else:
                print("No beds are available/Bed id is invalid")
                reward = self.config.get('invalid_action_penalty', -10) #potentially alter this penalty(lower)
        
        #BIG CHECK Advance time and process events 
        events = self._advance_time()
        self._process_events(events)
        
        # Update current patient if needed
        if self.current_patient is None and self.patient_queue:
            self.current_patient = self.patient_queue.pop(0)
        
        # Check for episode termination
        self.episode_steps += 1
        done = self.episode_steps >= self.max_steps
        
        # Get new state
        next_state = self.get_state()
        
        # Gather info
        info = self._get_info()
        
        return next_state, reward, done, info
    
    def get_state(self):
        """
        Returns:
            Dictionary containing patient features and available bed features
        """
        if self.current_patient:
            patient_features = self.current_patient.get_features()
        else:
            # No patient in queue - use zeros
            patient_features = [0] * self.config.get('patient_feature_dim', 5)

        available_beds = [bed for bed in self.beds if not bed.is_occupied()]
        bed_features = [bed.get_features() for bed in available_beds]

        #BIG BIG CHECK Pad or truncate bed features to fixed size if needed 
        max_beds = self.config['max_beds_in_state']
        if len(bed_features) < max_beds:
            # Pad with zeros
            pad_feature_dim = self.config.get('bed_feature_dim', 5)
            padding = [[0] * pad_feature_dim for _ in range(max_beds - len(bed_features))]
            bed_features.extend(padding)
        elif len(bed_features) > max_beds:
            # Truncate
            bed_features = bed_features[:max_beds]
        
        return {
            'patient': patient_features,
            'beds': bed_features
        }


    def _get_info(self):
        """
        Return additional information about current environment state.
        """
        return {
            'current_time': self.curr_time,
            'queue_length': len(self.patient_queue),
            'occupied_beds': sum(1 for bed in self.beds if bed.is_occupied()),
            'avg_wait_time': sum(self.stats['wait_times']) / max(1, len(self.stats['wait_times']))
        }
    
    #Environmental Helper Methods: 
    def _initialize_beds(self, beds_config):
        """Initialize bed objects based on configuration."""
        beds = [Bed(bed_id=i) for i in range(len(beds_config))]
        return beds


    def _handle_wait_action(self):
        """Process a wait action for the current patient."""
        if self.current_patient:
            # Calculate wait penalty based on severity
            # wait_penalty = -self.config.get('wait_penalty_factor', 0.1) * self.current_patient.severity
            self.current_patient.increase_wait_time()
            wait_penalty = -1 #subject to change
            return wait_penalty
        return 0

    def handle_assignment_action(self, bed_idx):
        """Process a bed assignment action."""
        if not self.current_patient:
            return self.config.get('invalid_action_penalty', -10)
        
        bed = self.beds[bed_idx]
        
        # Calculate reward components as per your document
        # Medical Match (MM)
        # specialty_match = 1 if self.current_patient.condition_type == bed.specialization else 0
        # mm_reward = 0 if specialty_match else -5  # -5 penalty for mismatch
        
        # Wait Time Consideration (WT)
        # wt_reward = -0.2 * self.current_patient.wait_time
        
        # Distance/Monitoring Penalty (DP)
        # dp_reward = -1 * (self.current_patient.severity / 10) * (1 / bed.monitoring_capability)
        
        # Total reward
        # total_reward = mm_reward + wt_reward + dp_reward
        total_reward = 0
        total_reward += self.calculate_reward(self.get_state(), len(self.patients))
        
        # Track stats
        # self.stats['wait_times'].append(self.current_patient.wait_time)
        # self.stats['specialty_matches'].append(specialty_match)
        
        # Assign patient to bed

        self.patients[self.current_patient.id] = self.current_patient
        bed.assign_patient(self.current_patient)
        
        # Clear current patient to get next one
        self.current_patient = None
        
        return total_reward
    
    def calculate_reward(self, state, occupancy, action=None, next_state=None): 
        """
        Returns reward 
        """
        if action == 0: 
            patient_features = state['patient']
            severity = patient_features[0]  # Assuming severity is first feature
            return -self.config.get('wait_penalty_factor', 0.1) * severity

        if 1 <= action <= len(self.beds):
            bed_idx = action - 1
            bed = self.beds[bed_idx]
            
            # patient_features = state['patient']
            # severity = patient_features[0]
            # condition = patient_features[2]  # Assuming condition is third feature
            # wait_time = patient_features[3]  # Assuming wait time is fourth feature
            
            # # Medical Match (MM)
            # mm = -5 if condition != bed.specialization else 0
            
            # # Wait Time Consideration (WT)
            # wt = -0.2 * wait_time
            
            # # Distance/Monitoring Penalty (DP)
            # dp = -1 * (severity / 10) * (1 / bed.monitoring_capability)
            
            # return mm + wt + dp
            return (occupancy / len(state['beds'])) * 3
        
        # Invalid action
        return self.config.get('invalid_action_penalty')

    def _advance_time(self):
        """
        Advance simulation time and return events that occurred.
        
        Returns:
            List of events (patient arrivals, discharges)
        """
        # Time step size - can be fixed or variable
        time_step = self.config.get('time_step')
        self.curr_time += time_step
        
        events = []
        
        # Check for new patient arrivals
        # Use time-dependent Poisson process
        # arrival_rate = self._get_arrival_rate(self.current_time)
        # num_arrivals = np.random.poisson(arrival_rate * time_step)
        
        # for _ in range(num_arrivals):
        #     events.append({
        #         'type': 'patient_arrival',
        #         'time': self.current_time
        #     })
        events.append({
            'type': 'patient_arrival',
            'time': self.curr_time
        })

        # Check for bed discharges
        for bed in self.beds:
            if bed.is_occupied() and bed.time_occupied >= bed.occupancy_delta:
                events.append({
                    'type': 'bed_discharge',
                    'bed_id': bed.bed_id,
                    'time': self.curr_time
                })
        
        return events

    def _process_events(self, events):
        """Process the events that occurred during time advancement."""
        for event in events:
            if event['type'] == 'patient_arrival':
                # new_patient = Patient.generate_random(self.current_time, self.config)
                new_patient = Patient(self.curr_time)
                self.patient_queue.append(new_patient)
            elif event['type'] == 'bed_discharge':
                bed_id = event['bed_id']

                # store deletion id because otherwise cannot reference patient because already gone
                deletion_id = self.beds[bed_id].current_patient.id

                self.beds[bed_id].discharge_patient()
                del self.patients[deletion_id]

    def _get_arrival_rate(self, time):
        """
        Get patient arrival rate based on time of day.
        
        Args:
            time: Current simulation time
            
        Returns:
            Arrival rate (patients per time unit)
        """
        # Implement time-dependent arrival rates
        # Example: Higher rates during day, lower at night
        # base_rate = self.config.get('base_arrival_rate', 0.1)
        
        # # Assume time is in hours and create daily cycle
        # hour_of_day = (time % 24)
        
        # # More arrivals during day (8am-8pm)
        # if 8 <= hour_of_day <= 20:
        #     return base_rate * 1.5
        # else:
        #     return base_rate * 0.5
    
        base_rate = self.config.get("base_arrival_rate")
        return base_rate
    
