import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
RGB_DIR = os.path.join(DATA_DIR, 'rgb')
XYZ_DIR = os.path.join(DATA_DIR, 'xyz')
BBOX_FILE = os.path.join(DATA_DIR, 'bboxes_light.csv')

# Define colors for visualization
COLORS = {
    'ego': 'blue',
    'traffic_light': 'yellow',
    'golf_cart': 'green',
    'barrel': 'red',
    'pedestrian': 'purple'
}

def load_traffic_light_bboxes():
    """Load traffic light bounding boxes from CSV file."""
    if not os.path.exists(BBOX_FILE):
        print(f"Warning: Bounding box file not found at {BBOX_FILE}")
        return None
    
    return pd.read_csv(BBOX_FILE)

def get_traffic_light_center(bbox_row):
    """Calculate the center of the traffic light bounding box."""
    x_min, y_min, x_max, y_max = bbox_row['x_min'], bbox_row['y_min'], bbox_row['x_max'], bbox_row['y_max']
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return int(center_x), int(center_y)

def get_3d_position(frame_id, u, v):
    """Get 3D position from depth data for a given pixel."""
    xyz_file = os.path.join(XYZ_DIR, f"frame_{frame_id:04d}.npz")
    
    if not os.path.exists(xyz_file):
        print(f"Warning: XYZ file not found at {xyz_file}")
        return None
    
    try:
        xyz = np.load(xyz_file)["points"]  # shape (H, W, 3)
        X, Y, Z = xyz[v, u]  # meters in camera coordinates
        
        # Check for invalid values
        if np.isnan(X) or np.isnan(Y) or np.isnan(Z) or np.isinf(X) or np.isinf(Y) or np.isinf(Z):
            return None
            
        return np.array([X, Y, Z])
    except Exception as e:
        print(f"Error loading 3D position for frame {frame_id}, pixel ({u}, {v}): {e}")
        return None

def define_ground_frame(traffic_light_3d):
    """Define ground frame with origin under the traffic light."""
    # The origin is directly under the traffic light on the ground
    # Z-axis passes upward through the traffic light
    # X-axis is forward, Y-axis is left (right-handed coordinate system)
    
    # Project traffic light position to the ground (Z=0)
    ground_position = np.array([traffic_light_3d[0], traffic_light_3d[1], 0])
    
    return ground_position

def compute_ego_trajectory(bboxes):
    """Compute ego-vehicle trajectory in ground frame."""
    trajectory = []
    traffic_light_3d_positions = []
    
    # Process each frame
    for row in bboxes:
        frame_id = int(row['frame_id'])
        u, v = get_traffic_light_center(row)
        
        # Get 3D position of traffic light
        traffic_light_3d = get_3d_position(frame_id, u, v)
        if traffic_light_3d is not None:
            traffic_light_3d_positions.append(traffic_light_3d)
    
    # Define ground frame based on first frame
    if not traffic_light_3d_positions:
        print("Error: No valid traffic light positions found")
        return None
    
    initial_traffic_light = traffic_light_3d_positions[0]
    ground_origin = define_ground_frame(initial_traffic_light)
    
    # Calculate ego trajectory relative to ground frame
    for i, traffic_light_pos in enumerate(traffic_light_3d_positions):
        # The ego position is the negative of the traffic light position
        # (since we're seeing the traffic light from the ego vehicle)
        ego_x = -traffic_light_pos[0] + initial_traffic_light[0]
        ego_y = -traffic_light_pos[1] + initial_traffic_light[1]
        
        # Add to trajectory
        trajectory.append((ego_x, ego_y))
    
    return trajectory

def detect_golf_cart(frame, frame_id):
    """Detect golf cart in the RGB frame using color thresholding."""
    # Load RGB image
    rgb_file = os.path.join(RGB_DIR, f"frame_{frame_id:04d}.png")
    if not os.path.exists(rgb_file):
        return None
    
    image = cv2.imread(rgb_file)
    if image is None:
        return None
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for white/light color (golf cart)
    # These values may need adjustment based on the actual appearance
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    
    # Create mask and apply it
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 1000  # Minimum area to consider
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not large_contours:
        return None
    
    # Get the largest contour (assuming it's the golf cart)
    largest_contour = max(large_contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate center
    center_x = x + w // 2
    center_y = y + h // 2
    
    return (center_x, center_y)

def detect_barrels(frame, frame_id):
    """Detect barrels in the RGB frame using color thresholding."""
    # Load RGB image
    rgb_file = os.path.join(RGB_DIR, f"frame_{frame_id:04d}.png")
    if not os.path.exists(rgb_file):
        return []
    
    image = cv2.imread(rgb_file)
    if image is None:
        return []
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for orange/red color (barrels)
    # These values may need adjustment based on the actual appearance
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # Create mask and apply it
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 100  # Minimum area to consider
    max_area = 5000  # Maximum area to consider
    barrel_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    barrel_centers = []
    for contour in barrel_contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center
        center_x = x + w // 2
        center_y = y + h // 2
        
        barrel_centers.append((center_x, center_y))
    
    return barrel_centers

def detect_pedestrians(frame, frame_id):
    """Detect pedestrians using a simple approach (placeholder)."""
    # This is a simplified placeholder function
    # In a real implementation, you would use a pedestrian detector like HOG or a neural network
    return []

def track_objects_in_frame(frame_id, xyz_data):
    """Track objects in a single frame and return their 3D positions."""
    objects = {
        'golf_cart': None,
        'barrels': [],
        'pedestrians': []
    }
    
    # Detect golf cart
    golf_cart_center = detect_golf_cart(None, frame_id)
    if golf_cart_center:
        u, v = golf_cart_center
        pos_3d = get_3d_position(frame_id, u, v)
        if pos_3d is not None:
            objects['golf_cart'] = pos_3d
    
    # Detect barrels
    barrel_centers = detect_barrels(None, frame_id)
    for center in barrel_centers:
        u, v = center
        pos_3d = get_3d_position(frame_id, u, v)
        if pos_3d is not None:
            objects['barrels'].append(pos_3d)
    
    # Detect pedestrians
    pedestrian_centers = detect_pedestrians(None, frame_id)
    for center in pedestrian_centers:
        u, v = center
        pos_3d = get_3d_position(frame_id, u, v)
        if pos_3d is not None:
            objects['pedestrians'].append(pos_3d)
    
    return objects

def create_enhanced_bev_animation(trajectory, bboxes, output_file="enhanced_trajectory.mp4"):
    """Create an enhanced BEV animation with additional objects."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set up the plot
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Enhanced Bird\'s-Eye View with Object Tracking')
    ax.grid(True)
    
    # Mark traffic light position (origin)
    traffic_light_point = ax.scatter(0, 0, c=COLORS['traffic_light'], s=100, marker='*', label='Traffic Light')
    
    # Initialize plot elements
    ego_line, = ax.plot([], [], c=COLORS['ego'], linewidth=2)
    ego_point = ax.scatter([], [], c=COLORS['ego'], s=100, marker='o', label='Ego Vehicle')
    
    # Initialize objects
    golf_cart_point = ax.scatter([], [], c=COLORS['golf_cart'], s=100, marker='s', label='Golf Cart')
    barrel_points = ax.scatter([], [], c=COLORS['barrel'], s=50, marker='^', label='Barrels')
    pedestrian_points = ax.scatter([], [], c=COLORS['pedestrian'], s=50, marker='d', label='Pedestrians')
    
    # Set axis limits based on trajectory
    x_coords = [point[0] for point in trajectory]
    y_coords = [point[1] for point in trajectory]
    margin = 5  # meters
    ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    ax.set_aspect('equal')
    ax.legend()
    
    # Get initial traffic light position
    initial_frame = int(bboxes.iloc[0]['frame_id'])
    u, v = get_traffic_light_center(bboxes.iloc[0])
    initial_traffic_light_pos = get_3d_position(initial_frame, u, v)
    
    # Track objects for each frame
    all_objects = []
    for i, (_, row) in enumerate(bboxes.iterrows()):
        frame_id = int(row['frame_id'])
        
        # Load XYZ data
        xyz_file = os.path.join(XYZ_DIR, f"frame_{frame_id:04d}.npz")
        if not os.path.exists(xyz_file):
            all_objects.append(None)
            continue
            
        try:
            xyz_data = np.load(xyz_file)["points"]
            objects = track_objects_in_frame(frame_id, xyz_data)
            
            # Transform to ground frame
            if initial_traffic_light_pos is not None:
                if objects['golf_cart'] is not None:
                    objects['golf_cart'][0] = -objects['golf_cart'][0] + initial_traffic_light_pos[0]
                    objects['golf_cart'][1] = -objects['golf_cart'][1] + initial_traffic_light_pos[1]
                
                for i in range(len(objects['barrels'])):
                    objects['barrels'][i][0] = -objects['barrels'][i][0] + initial_traffic_light_pos[0]
                    objects['barrels'][i][1] = -objects['barrels'][i][1] + initial_traffic_light_pos[1]
                
                for i in range(len(objects['pedestrians'])):
                    objects['pedestrians'][i][0] = -objects['pedestrians'][i][0] + initial_traffic_light_pos[0]
                    objects['pedestrians'][i][1] = -objects['pedestrians'][i][1] + initial_traffic_light_pos[1]
            
            all_objects.append(objects)
        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")
            all_objects.append(None)
    
    def init():
        ego_line.set_data([], [])
        ego_point.set_offsets(np.empty((0, 2)))
        golf_cart_point.set_offsets(np.empty((0, 2)))
        barrel_points.set_offsets(np.empty((0, 2)))
        pedestrian_points.set_offsets(np.empty((0, 2)))
        return ego_line, ego_point, golf_cart_point, barrel_points, pedestrian_points
    
    def update(frame):
        # Update ego trajectory
        x_data = [point[0] for point in trajectory[:frame+1]]
        y_data = [point[1] for point in trajectory[:frame+1]]
        ego_line.set_data(x_data, y_data)
        
        if frame < len(trajectory):
            ego_point.set_offsets(np.array([[x_data[-1], y_data[-1]]]))
        
        # Update objects
        if frame < len(all_objects) and all_objects[frame] is not None:
            objects = all_objects[frame]
            
            # Update golf cart
            if objects['golf_cart'] is not None:
                golf_cart_point.set_offsets(np.array([[objects['golf_cart'][0], objects['golf_cart'][1]]]))
            else:
                golf_cart_point.set_offsets(np.empty((0, 2)))
            
            # Update barrels
            if objects['barrels']:
                barrel_x = [barrel[0] for barrel in objects['barrels']]
                barrel_y = [barrel[1] for barrel in objects['barrels']]
                barrel_points.set_offsets(np.column_stack((barrel_x, barrel_y)))
            else:
                barrel_points.set_offsets(np.empty((0, 2)))
            
            # Update pedestrians
            if objects['pedestrians']:
                ped_x = [ped[0] for ped in objects['pedestrians']]
                ped_y = [ped[1] for ped in objects['pedestrians']]
                pedestrian_points.set_offsets(np.column_stack((ped_x, ped_y)))
            else:
                pedestrian_points.set_offsets(np.empty((0, 2)))
        
        return ego_line, ego_point, golf_cart_point, barrel_points, pedestrian_points
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(trajectory),
                         init_func=init, blit=True, interval=100)
    
    # Save animation
    anim.save(output_file, writer='ffmpeg', fps=10)
    print(f"Enhanced trajectory animation saved to {output_file}")

def main():
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        print("Please download the dataset from the provided link and extract it to the 'dataset' folder.")
        return
    
    # Load traffic light bounding boxes
    bboxes = load_traffic_light_bboxes()
    if bboxes is None:
        return
    
    # Compute ego trajectory
    trajectory = compute_ego_trajectory(bboxes)
    if trajectory is None:
        return
    
    # Create enhanced BEV animation
    try:
        create_enhanced_bev_animation(trajectory, bboxes, "enhanced_trajectory.mp4")
    except Exception as e:
        print(f"Warning: Could not create enhanced animation: {e}")
        print("You may need to install ffmpeg for video creation.")

if __name__ == "__main__":
    main()