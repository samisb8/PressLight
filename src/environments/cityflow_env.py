import cityflow
import json
import numpy as np
import os
import math
from typing import Dict, List, Tuple

class CityFlowPressLightEnv:
    """
    A CityFlow Environment Wrapper for PressLight it takes the complex CityFlow simulator and turns it into a 
    standard OpenAI Gym-style interface (reset and step) that an RL agent can understand the agent will work 
    with two buttons reset() and step(action) and it doesn't now nothing about how to restart a training session
    (choose cars put them at the beggining etc) and also when it takes action it doesn't know to realize it in 
    reality it is for our wrapper to handle this details
    """

    def __init__(self, 
                 roadnet_file: str, 
                 flow_file: str, 
                 intersection_id: str,
                 num_steps: int = 4000, 
                 step_len: int = 10):  #we chose step_len = 10. Let's analyze why you didn't choose 1.
                                       #If the agent chooses "Green for East-West," and you only simulate 
                                       #for 1 second: The light turns green.The cars haven't even had time to 
                                       #accelerate (physics).The "State" hasn't changed.The "Reward" is still bad.
        
        # handle paths. it is to be sure it will work wether i used it from the src or the root folder 
        # it will always work
        self.roadnet_file_abs = os.path.abspath(roadnet_file)
        self.flow_file_abs = os.path.abspath(flow_file)
        
        self.data_dir = os.path.dirname(self.roadnet_file_abs)
        self.roadnet_filename = os.path.basename(self.roadnet_file_abs)
        self.flow_filename = os.path.basename(self.flow_file_abs)
        
        if not os.path.exists(self.roadnet_file_abs):
            raise FileNotFoundError(f"Roadnet file not found: {self.roadnet_file_abs}")

        self.intersection_id = intersection_id
        self.num_steps = num_steps
        self.step_len = step_len
        self.current_step = 0
        self.current_phase_index = 0

        # Generate a temporary config file for CityFlow
        self.config_file = self._generate_config_file()  # see the explanation of this method below
        self.engine = cityflow.Engine(self.config_file, thread_num=1)
        
        # Parse Roadnet to get Lane Information
        self._parse_roadnet()
        
        # Define dimensions
        self.num_phases = len(self.phases)
        
        # State Dim = Phase_OneHot + (Incoming_Lanes * 3_Segments) + Outgoing_Lanes: see the paper presslight for more details
        self.state_dim = (
            self.num_phases + 
            (len(self.incoming_lanes) * 3) + 
            len(self.outgoing_lanes)
        )

    def _generate_config_file(self) -> str:
        """Creates a config.json pointing to the correct data directory."""
        config = {
            "interval": 1.0,
            "seed": 0,
            "dir": self.data_dir + "/", 
            "roadnetFile": self.roadnet_filename,
            "flowFile": self.flow_filename,
            "rlTrafficLight": True, #it tells CityFlow: "Don't use your internal 
                                    #timers; wait for my Python code to tell you when to change lights
            "saveReplay": False,
            "roadnetLogFile": "roadnet_log.json",
            "replayLogFile": "replay_log.txt"
        }
        config_path = os.path.abspath("temp_cityflow_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        return config_path

    def _parse_roadnet(self):
        """a method tht will analyzes the roadnet JSON file in order to identify lanes:  CityFlow's roadnet.json 
        is a list of roads, but it doesn't explicitly tell you "these lanes are for this intersection." 
        You have to find them."""
        with open(self.roadnet_file_abs, "r") as f:
            roadnet = json.load(f)
        #they are looking for the specific "node" in the map. Once found, you extract the lightphases. 
        #These define the Action Space. If there are 8 phases, your agent has 8 buttons it can press.
        inter_data = next((i for i in roadnet['intersections'] if i['id'] == self.intersection_id), None)
        if not inter_data:
            raise ValueError(f"Intersection {self.intersection_id} not found!")

        self.phases = inter_data['trafficLight']['lightphases']
        self.incoming_lanes = []
        self.outgoing_lanes = []
        self.lane_lengths = {} 

        for road in roadnet['roads']:
            road_id = road['id']
            # If road ends at our intersection -> Incoming
            if road['endIntersection'] == self.intersection_id:
                for i in range(len(road['lanes'])):
                    lane_id = f"{road_id}_{i}"
                    self.incoming_lanes.append(lane_id)
                    
                    points = road['points']
                    p1, p2 = points[0], points[1]
                    dist = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2) #store this so we can later divide 
                                                                                      #the road into 3 segments.
                    self.lane_lengths[lane_id] = dist

            # If road starts at our intersection -> Outgoing
            elif road['startIntersection'] == self.intersection_id:
                for i in range(len(road['lanes'])):
                    lane_id = f"{road_id}_{i}"
                    self.outgoing_lanes.append(lane_id)

        print(f"Parsed {self.intersection_id}: {len(self.incoming_lanes)} incoming, {len(self.outgoing_lanes)} outgoing lanes.")

    def reset(self) -> np.ndarray:
        self.engine.reset()
        self.current_step = 0
        self.current_phase_index = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """The agent gives us an integer (0, 1, 2, or 3). We tell CityFlow: "Switch the light to this phase."""
        if action >= self.num_phases or action < 0:
             action = 0

        self.current_phase_index = action
        self.engine.set_tl_phase(self.intersection_id, action)
        #In traffic, nothing happens in 1 second. We run the simulation for 10 seconds (step_len) before looking 
        #again. This prevents the agent from flickering the lights every second, which will be impossible 
        #in the real world.
        for _ in range(self.step_len):  
            self.engine.next_step()
            self.current_step += 1
            if self.current_step >= self.num_steps:
                break

        state = self._get_state()
        reward = self._get_reward()
        done = self.current_step >= self.num_steps
        
        return state, reward, done, {}

    def _get_state(self) -> np.ndarray:
        """
        a method to build the "Observation" vector: [Phase_OneHot, Incoming_Segments, Outgoing_Lanes]
        """
        # 1. One-Hot Phase
        phase_one_hot = np.zeros(self.num_phases)
        if self.num_phases > 0:
            phase_one_hot[self.current_phase_index] = 1
        
        # 2. Vehicle Counts
        lane_vehicle_counts = self.engine.get_lane_vehicle_count()
        vehicle_ids = self.engine.get_vehicles()
        
        # Map: lane_id -> [seg1, seg2, seg3]
        incoming_counts = {lane: [0, 0, 0] for lane in self.incoming_lanes}
        
        for v_id in vehicle_ids:
            try: 
                #Every second, you ask the engine for a list of all active car IDs. For each car, 
                #you get a dictionary. You specifically look for:drivable: The ID of the lane the car is on.
                #distance: How many meters the car has traveled from the start of that lane.
                info = self.engine.get_vehicle_info(v_id)
            except:
                continue

            if 'drivable' in info:
                lane_full_id = info['drivable']
            elif 'road' in info:
                lane_full_id = f"{info['road']}_{info.get('lane', 0)}"
            else:
                continue
            
            # Check if this vehicle is on an incoming lane
            if lane_full_id in self.incoming_lanes:
                distance = float(info['distance'])
                length = self.lane_lengths[lane_full_id]
                #A car 5 meters from the light is much more important for immediate pressure than a car 
                # 200 meters away. By dividing the lane into 3 segments, we give the agent "Depth Perception."

                if distance < length / 3:
                    seg_idx = 2 # Far
                elif distance < 2 * length / 3:
                    seg_idx = 1 # Middle
                else:
                    seg_idx = 0 # Near
                
                incoming_counts[lane_full_id][seg_idx] += 1

        # Flatten Incoming Segments
        incoming_vector = []
        for lane in self.incoming_lanes:
            incoming_vector.extend(incoming_counts[lane])

        # Outgoing Lanes
        outgoing_vector = []
        for lane in self.outgoing_lanes:
            count = lane_vehicle_counts.get(lane, 0)
            outgoing_vector.append(count)
            
        state_concat = np.concatenate([phase_one_hot, incoming_vector, outgoing_vector])
    
    # Normalize! 
    # We assume max capacity of a segment is roughly 30 vehicles.
    # This keeps values between 0 and 1, making the Neural Network learn much faster.
        norm_factor = 50.0 
        return state_concat / norm_factor

    def _get_reward(self) -> float:
        """Reward = -Pressure = -|Sum(Incoming) - Sum(Outgoing)| as it was introduced in the paper"""
        lane_vehicle_counts = self.engine.get_lane_vehicle_count()
        
        incoming_sum = 0
        for lane in self.incoming_lanes:
            incoming_sum += lane_vehicle_counts.get(lane, 0)
            
        outgoing_sum = 0
        for lane in self.outgoing_lanes:
            outgoing_sum += lane_vehicle_counts.get(lane, 0)
            
        return -1.0 * abs(incoming_sum - outgoing_sum)/100.0

    @property
    def action_space_n(self):
        return self.num_phases


#test:

if __name__ == "__main__":
    current_dir = os.getcwd()
    roadnet_path = os.path.join(current_dir, "data", "NewYork", "roadnet_16_1.json")
    flow_path = os.path.join(current_dir, "data", "NewYork", "anon_16_1_300_newyork_real_1.json")
    
    test_inter_id = "intersection_1_1" 

    try:
        print(f"Testing with Roadnet: {roadnet_path}")
        print("Initializing Environment...")
        
        env = CityFlowPressLightEnv(roadnet_path, flow_path, test_inter_id)
        
        print(f"State Dimension: {env.state_dim}")
        print(f"Number of Phases: {env.num_phases}")
        
        obs = env.reset()
        print(f"Initial State shape: {obs.shape}")
        
        print("Running simulation steps...")
        for i in range(5):
            action = np.random.randint(0, env.num_phases)
            obs, reward, done, _ = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, StateSum={np.sum(obs)}")
            
        print("Environment Test Passed Successfully.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()