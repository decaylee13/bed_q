from bed import Bed
from patient import Patient
class HospitalBedEnv(): 
    def __init__(self, config):
        self.config = config.get('environment_parameters')

        self.curr_time = 0 
        self.curr_episode = 0
        self.episode_steps = 0 
        self.max_steps = self.config.get('max_episode_steps', 1000)

        self.patients = {} 
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
        self.episode_steps = 0 

        for beds in self.beds: #check
            beds.reset() 

        self.patients = {}
        self.patient_queue = []
        self.current_patient = Patient(self.curr_time)

        return self.get_state()#check

    def step(self, action):
        """Process an action and return the next state, reward, and done flag."""
        bed_idx = action - 1
        invalid_action = False
        
        if action == 0:  
            # Wait action
            self._handle_wait_action()
            reward = self.calculate_reward(occupancy=len(self.patients), action=action, bed_idx=bed_idx)
        else:
            # Check if this is a valid bed assignment
            if bed_idx < len(self.beds) and not self.beds[bed_idx].is_occupied():
                reward = self.calculate_reward(occupancy=len(self.patients) + 1, action=action, bed_idx=bed_idx)
                self.handle_assignment_action(bed_idx, action)
            else:
                # Invalid assignment - apply penalty
                invalid_action = True
                reward = self.calculate_reward(occupancy=len(self.patients), action=action, bed_idx=bed_idx)
                self.current_patient = None  # Move to next patient
        
        # Advance time and process events
        self._advance_time()
        
        # Update current patient if needed
        if self.current_patient is None and self.patient_queue:
            self.current_patient = self.patient_queue.pop(0)
        
        # Check for episode termination
        self.episode_steps += 1
        done = self.episode_steps >= self.max_steps
        
        # Get new state
        next_state = self.get_state()
        
        # Add information about valid actions to help the agent learn
        info = self._get_info()
        info['invalid_action'] = invalid_action
        
        return next_state, reward, done, info
    
    def get_state(self):
        # Get current patient features or empty vector if none
        if self.current_patient:
            patient_features = self.current_patient.get_features()
        else:
            patient_features = [0] * self.config.get('patient_feature_dim', 5)
        
        # Include ALL beds instead of just unoccupied ones
        bed_features = [bed.get_features() for bed in self.beds]
        
        # Limit to max_beds if needed
        max_beds = self.config['max_beds_in_state']
        if len(bed_features) > max_beds:
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
        beds = []
        total_beds = len(beds_config)
        
        low_count = int(total_beds * 0.7)
        medium_count = int(total_beds * 0.2)
        high_count = total_beds - low_count - medium_count

        for i in range(low_count):
            eff = 1 + (i%5)
            new_bed = Bed(bed_id = i, efficiency = eff)
            beds.append(new_bed)
        
        for i in range(low_count, low_count + medium_count):
            eff = 6 + (i % 3)
            new_bed = Bed(bed_id = i, efficiency = eff)
            beds.append(new_bed)

        for i in range(low_count + medium_count, total_beds):
            eff = 9 + (i % 2)
            new_bed = Bed(bed_id=i, efficiency=eff)
            beds.append(new_bed)
        
        return beds


    def _handle_wait_action(self):
        """Process a wait action for the current patient."""
        if self.current_patient:
            # Calculate wait penalty based on severity
            # wait_penalty = -self.config.get('wait_penalty_factor', 0.1) * self.current_patient.severity
            self.current_patient.increase_wait_time()
            # wait_penalty = -1 #subject to change
            # return wait_penalty
        # return 0

    def handle_assignment_action(self, bed_idx, action):
        """Process a bed assignment action."""
        if action != 0:
            bed = self.beds[bed_idx]
            self.patients[self.current_patient.id] = self.current_patient # reference the patient's existing ID
            bed.assign_patient(self.current_patient)
            # Clear current patient to get next one
            self.current_patient = None
    
    def calculate_reward(self, occupancy, action, bed_idx): 
        """
        Returns reward 
        """
        reward = 0 

        # if action!=0:
        #     print("is the bed of the desired action occupied?")
            # print("bed_idx's patient (if any)",self.beds[bed_idx].current_patient)
            # print(not (self.beds[bed_idx].is_occupied()))
    
        if action == 0: 
            # if making a patient wait
            reward -= self.config.get('wait_penalty_factor') 

        elif bed_idx < self.config.get('max_beds_in_state') and not self.beds[bed_idx].is_occupied():
            # if a valid action
            reward += self.config.get('valid_bed_asnmt')

            # update the occupancy reward
            reward += (occupancy / self.config.get('max_beds_in_state')) * 10
        else:
            # Discourage it from doing illegal actions
            reward -= self.config.get('invalid_action_penalty')
            # print("Incurred invalid action reward",self.curr_time)
        
        
        return reward

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

            if bed.is_occupied():
                bed.time_occupied_increase()

            if bed.is_occupied() and bed.time_occupied >= bed.occupancy_delta:
                events.append({
                    'type': 'bed_discharge',
                    'bed_id': bed.bed_id,
                    'time': self.curr_time
                })
        # print(events)
        for event in events:
            if event['type'] == 'patient_arrival':
                # new_patient = Patient.generate_random(self.current_time, self.config)
                new_patient = Patient(self.curr_time)
                self.patient_queue.append(new_patient)
            elif event['type'] == 'bed_discharge':
                bed_id = event['bed_id']

                # store deletion id because otherwise cannot reference patient because already gone
                deletion_id = self.beds[bed_id].current_patient.id
                # print(f"before: {[target.is_occupied() for target in self.beds]}")
                self.beds[bed_id].discharge_patient()
                del self.patients[deletion_id]
                # print(f"after: {[target.is_occupied() for target in self.beds]}")
                

    def _get_arrival_rate(self, time):
        """
        Get patient arrival rate based on time of day.
        
        Args:
            time: Current simulation time
            
        Returns:
            Arrival rate (patients per time unit)
        """
        base_rate = self.config.get("base_arrival_rate")
        return base_rate
    
    def get_valid_actions(self):
        """Return list of valid actions (0 for wait, 1-50 for unoccupied beds)"""
        valid_actions = [0]  # Wait is always valid
        
        for i, bed in enumerate(self.beds):
            if not bed.is_occupied():
                valid_actions.append(i+1)  # +1 because 0 is wait
        
        return valid_actions