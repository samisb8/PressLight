# src/algorithms/fixed_time.py
class FixedTimeAgent:
    """The traditional traffic signal controller (Fixed Cycle like 10 seconds for red and other 10 sec for geen)"""
    def __init__(self, num_phases: int, phase_duration: int = 30, step_len: int = 10):
        self.num_phases = num_phases  #see paper for definition of phase
        self.phase_duration = phase_duration
        self.step_len = step_len
        self.current_phase = 0
        self.time_in_current_phase = 0

    def get_action(self, state):
        self.time_in_current_phase += self.step_len
        if self.time_in_current_phase >= self.phase_duration:
            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.time_in_current_phase = 0
        return self.current_phase