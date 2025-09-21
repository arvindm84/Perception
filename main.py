import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
RGB_DIR = os.path.join(DATA_DIR, 'rgb')
XYZ_DIR = os.path.join(DATA_DIR, 'xyz')
BBOX_FILE = os.path.join(DATA_DIR, 'bboxes_light.csv')

def load_traffic_light_bboxes():
    if not os.path.exists(BBOX_FILE):
        print(f"Warning: Bounding box file not found at {BBOX_FILE}")
        return None
    
    result = []
    with open(BBOX_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            header_mapping = {'frame': 'frame_id', 'x1': 'x_min', 'y1': 'y_min', 'x2': 'x_max', 'y2': 'y_max'}
            row = {header_mapping[key]: float(value) for key, value in row.items() if key in header_mapping}
            result.append(row)
    return result

def get_traffic_light_center(bbox_row):
    x_min, y_min, x_max, y_max = bbox_row['x_min'], bbox_row['y_min'], bbox_row['x_max'], bbox_row['y_max']
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return int(center_x), int(center_y)

def get_3d_position(frame_id, u, v):
    xyz_file = os.path.join(XYZ_DIR, f"depth{frame_id:06d}.npz")
    
    if not os.path.exists(xyz_file):
        print(f"Warning: XYZ file not found at {xyz_file}")
        return None
    
    try:
        xyz = np.load(xyz_file)["xyz"]
        point = xyz[v, u]
        
        X, Y, Z = point[:3]
        
        if np.isnan(X) or np.isnan(Y) or np.isnan(Z) or np.isinf(X) or np.isinf(Y) or np.isinf(Z):
            return None
            
        return np.array([X, Y, Z])
    except Exception as e:
        print(f"Error loading 3D position for frame {frame_id}, pixel ({u}, {v}): {e}")
        return None

def define_ground_frame(traffic_light_3d):
    ground_position = np.array([traffic_light_3d[0], traffic_light_3d[1], 0])
    return ground_position

def compute_ego_trajectory(bboxes):
    trajectory = []
    traffic_light_3d_positions = []
    frame_ids = []
    
    for row in tqdm(bboxes, total=len(bboxes), desc="Processing frames"):
        frame_id = int(row['frame_id'])
        u, v = get_traffic_light_center(row)
        
        traffic_light_3d = get_3d_position(frame_id, u, v)
        if traffic_light_3d is not None:
            traffic_light_3d_positions.append(traffic_light_3d)
            frame_ids.append(frame_id)
    
    if not traffic_light_3d_positions:
        print("Error: No valid traffic light positions found")
        return None
    
    initial_traffic_light = traffic_light_3d_positions[0]
    ground_origin = define_ground_frame(initial_traffic_light)
    
    for i, traffic_light_pos in enumerate(traffic_light_3d_positions):
        ego_x = -traffic_light_pos[0] + initial_traffic_light[0]
        ego_y = -traffic_light_pos[1] + initial_traffic_light[1]
        
        trajectory.append((ego_x, ego_y, frame_ids[i]))
    
    return trajectory

def plot_trajectory(trajectory, output_file="trajectory.png"):
    plt.figure(figsize=(10, 8))
    
    x_coords = [point[0] for point in trajectory]
    y_coords = [point[1] for point in trajectory]
    
    plt.plot(x_coords, y_coords, 'b-', linewidth=2)
    plt.scatter(x_coords, y_coords, c='blue', s=10)
    
    plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='^', label='Start')
    plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='o', label='End')
    
    plt.scatter(0, 0, c='yellow', s=100, marker='*', label='Traffic Light')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Ego-Vehicle Trajectory in Bird\'s-Eye View')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    plt.savefig(output_file, dpi=300)
    print(f"Trajectory plot saved to {output_file}")
    
    return plt.gcf()

def create_trajectory_animation(trajectory, output_file="trajectory.mp4"):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Ego-Vehicle Trajectory in Bird\'s-Eye View')
    ax.grid(True)
    
    ax.scatter(0, 0, c='yellow', s=100, marker='*', label='Traffic Light')
    
    x_coords = [point[0] for point in trajectory]
    y_coords = [point[1] for point in trajectory]
    margin = 2
    ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    ax.set_aspect('equal')
    ax.legend()
    
    line, = ax.plot([], [], 'b-', linewidth=2)
    points = ax.scatter([], [], c='blue', s=10)
    current_point = ax.scatter([], [], c='red', s=100, marker='o')
    
    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        points.set_offsets(np.empty((0, 2)))
        current_point.set_offsets(np.empty((0, 2)))
        frame_text.set_text('')
        return line, points, current_point, frame_text
    
    def update(frame):
        x_data = [point[0] for point in trajectory[:frame+1]]
        y_data = [point[1] for point in trajectory[:frame+1]]
        line.set_data(x_data, y_data)
        
        if frame > 0:
            points.set_offsets(np.column_stack((x_data[:-1], y_data[:-1])))
        
        if frame < len(trajectory):
            current_point.set_offsets(np.array([[x_data[-1], y_data[-1]]]))
            frame_text.set_text(f'Frame: {trajectory[frame][2]}')
        
        return line, points, current_point, frame_text
    
    anim = FuncAnimation(fig, update, frames=len(trajectory),
                         init_func=init, blit=True, interval=100)
    
    anim.save(output_file, writer='ffmpeg', fps=10, dpi=200)
    print(f"Trajectory animation saved to {output_file}")

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        print("Please download the dataset from the provided link and extract it to the 'dataset' folder.")
        return
    
    bboxes = load_traffic_light_bboxes()
    if bboxes is None:
        return
    
    trajectory = compute_ego_trajectory(bboxes)
    if trajectory is None:
        return
    
    plot_trajectory(trajectory)
    
    try:
        create_trajectory_animation(trajectory)
    except Exception as e:
        print(f"Warning: Could not create animation: {e}")
        print("You may need to install ffmpeg for video creation.")

if __name__ == "__main__":
    main()