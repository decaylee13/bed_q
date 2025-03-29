
class HospitalBedEnv(): 
    def __init__(self, config):
        self.config = config 

        self.curr_time = 0 
        self.curr_episode = 0
        self.max_steps = config.get('max_episode_steps', 1000)

        self.patients = [] #check
        self.beds = self._initialize_beds(config['beds_config'])
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

        for beds in self.beds: #check
            beds.reset() 

        self.patients = []
        self.patient_queue = []
        self._generate_initial_patients()#check 

        return self._get_state()#check
    
    def step(self, action):
        """
        Process an action and return the next state, reward, and done flag.
        """
        reward = 0
        
        if action == 0:  
            reward = self._handle_wait_action() #check
        else: 
            bed_idx = action - 1 #bed id from 0 to N
            if bed_idx < len(self.beds) and self.beds[bed_idx].is_available():
                reward = self._handle_assignment_action(bed_idx)
            else:
                print("No beds are available/Bed id is invalid")
                reward = self.config.get('invalid_action_penalty', -10) #check
        
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
        next_state = self._get_state()
        
        # Gather info
        info = self._get_info()
        
        return next_state, reward, done, info
    
    def __getstate__(self):
        """
        Returns:
            Dictionary containing patient features and available bed features
        """
        if self.current_patient:
            patient_features = self.current_patient.get_features()
        else:
            # No patient in queue - use zeros
            patient_features = [0] * self.config.get('patient_feature_dim', 5)

        available_beds = [bed for bed in self.beds if bed.is_available()]
        bed_features = [bed.get_features() for bed in available_beds]

        # Pad or truncate bed features to fixed size if needed
        max_beds = self.config.get('max_beds_in_state', 10)
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
            'current_time': self.current_time,
            'queue_length': len(self.patient_queue),
            'occupied_beds': sum(1 for bed in self.beds if not bed.is_available()),
            'avg_wait_time': sum(self.stats['wait_times']) / max(1, len(self.stats['wait_times']))
        }
