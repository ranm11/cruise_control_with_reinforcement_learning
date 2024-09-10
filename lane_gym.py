import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

class CarLaneTrackingEnv(gym.Env):
    def __init__(self):
        super(CarLaneTrackingEnv, self).__init__()
        
        # Action space: Wheel range [-1, 1] where -1 is full left and 1 is full right
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State space: [theta, theta_dot, car_speed, distance_from_lane]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, 0.0, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, 100.0, np.inf], dtype=np.float32)
        )
        
        # Initialize state
        self.state = None
        self.time_step = 0.1  # Time step for the simulation
        self.lane_curvature = 0.0  # Initial curvature of the lane
        self.wheel_range = 0
        # Matplotlib setup
        self.fig, self.ax = plt.subplots()
        self.lane_width = 2.0  # Lane width in meters
        self.car_length = 1.0  # Car length in meters
        self.car_width = 0.5  # Car width in meters
        self.plot_initialized = False
        
    def reset(self):
        # Reset the state to initial conditions
        theta = np.random.uniform(-0.1, 0.1)  # Small initial angle deviation
        theta_dot = 0.0  # Initially no rotation
        car_speed = 10.0  # Constant speed of 20 m/s
        distance_from_lane = np.random.uniform(-1.0, 1.0)  # Initial distance from lane (e.g., -1 to 1 meter)
        
        self.state = np.array([theta, theta_dot, car_speed, distance_from_lane], dtype=np.float32)
        self._init_plot()
        return self.state
    
    def SpeedUpdate(self,new_speed):
        self.state[2] = new_speed

    def step(self, action):
        theta, theta_dot, car_speed, distance_from_lane = self.state
        
        # Action is the wheel range input
        self.wheel_range = action[0]
        
        # Update lane curvature randomly to simulate curves
        self.lane_curvature += np.random.uniform(-0.01, 0.01)
        self.lane_curvature = np.clip(self.lane_curvature, -0.05, 0.05)  # Limit the curvature to a small range
        
        # Simplified car dynamics
        # Update theta_dot based on wheel input and lane curvature
        theta_dot += self.wheel_range * car_speed * 0.05 - self.lane_curvature
        
        # Update theta based on theta_dot
        theta += theta_dot * self.time_step
        
        # Normalize theta to stay within [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        # Update distance from lane based on theta, speed, and lane curvature
        distance_from_lane += np.tan(theta) * car_speed * self.time_step + self.lane_curvature * car_speed * self.time_step
        
        # Update state
        self.state = np.array([theta, theta_dot, car_speed, distance_from_lane], dtype=np.float32)
        
        # Example reward: Penalize deviation from lane and large theta
        reward = -np.abs(distance_from_lane) - np.abs(theta)*3
        
        # Example done condition: Episode ends if car speed is 0 (in case of more advanced modeling)
        done = np.abs(distance_from_lane) > 1
        
        return self.state, reward, done, {}
    
    def render(self, mode='human'):
        theta, theta_dot, car_speed, distance_from_lane = self.state
        
        # Clear previous plot
        self.ax.clear()
        
        # Draw lane with a curve based on the current lane curvature
        lane_y = np.linspace(-25, 25, 100)
        lane_x = self.lane_curvature * (lane_y**2)  # Quadratic curve for lane
        
        self.ax.plot(lane_x, lane_y, 'k--')  # Dashed line for lane center
        
        # Draw lane boundaries
        self.ax.plot(lane_x - self.lane_width / 2, lane_y, 'k-')
        self.ax.plot(lane_x + self.lane_width / 2, lane_y, 'k-')
        
        # Draw car as a rotated rectangle
        car_y = distance_from_lane
        car_x = 0  # Car is always centered horizontally in the view
        car = patches.Rectangle((car_x - self.car_width / 2, car_y - self.car_length / 2), self.car_width, self.car_length, edgecolor='blue', facecolor='blue', alpha=0.8)
        
        # Create the transformation for rotation and translation
        transform = Affine2D().rotate_deg_around(car_x, car_y, np.degrees(theta)) + self.ax.transData
        car.set_transform(transform)
        
        self.ax.add_patch(car)
        
        # Set plot limits and labels
        self.ax.set_xlim(-25, 25)
        self.ax.set_ylim(-25, 25)
        self.ax.set_xlabel('Car Position')
        self.ax.set_ylabel('Distance from Lane Center (m)')
        
        # Display the state on the plot
        state_text = (
            f"Wheel Steering: {self.wheel_range:.2f}\n"
            f"Theta: {theta:.2f}\n"
            f"Theta Dot: {theta_dot:.2f}\n"
            f"Speed: {3.6*car_speed:.2f} Km/h\n"
            f"Distance from Lane: {distance_from_lane:.2f} m"
        )
        self.ax.text(0.05, 0.95, state_text, transform=self.ax.transAxes,
                     fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # Update plot
        plt.draw()
        plt.pause(0.001)
    
    def close(self):
        plt.close()
    
    def _init_plot(self):
        if not self.plot_initialized:
            plt.ion()
            self.plot_initialized = True
            self.render()

StandAlone = False
if(StandAlone):
# Example usage
    env = CarLaneTrackingEnv()
    num_states = env.observation_space.shape[0]
    state = env.reset()

    for _ in range(700):
        action = env.action_space.sample()  # Random action for testing
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            #break
            env.reset()

    env.close()
