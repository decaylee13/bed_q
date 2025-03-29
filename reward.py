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