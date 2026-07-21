import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch
from matplotlib import rcParams
from sklearn.neighbors import BallTree
rcParams['axes.grid'] = True
rcParams['font.size'] = 18
from shapely.geometry import Point, Polygon, LineString
import random

###### Generate polygons
# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------
def generate_circles(occupancy_rate, xmin, xmax, ymin, ymax, r_min, r_max, 
                     start_point=None, end_point=None, safe_radius=0.0):
    """
    Generates a list of circles (as tuples of (x, y, r)) based on target occupancy rate.
    
    Args:
        occupancy_rate: float (0.0 to 1.0), fraction of workspace to occupy.
        xmin, xmax, ymin, ymax: workspace boundaries.
        r_min, r_max: radius constraints for obstacles.
        start_point, end_point: (x, y) coordinates.
        safe_radius: distance to keep clear around start/end points.
    """
    workspace_area = (xmax - xmin) * (ymax - ymin)
    target_area = occupancy_rate * workspace_area
    current_area = 0.0
    circles = []
    
    tries = 0
    while current_area < target_area and tries < 20000:
        tries += 1
        r = random.uniform(r_min, r_max)
        center = [random.uniform(xmin + r, xmax - r),
                  random.uniform(ymin + r, ymax - r)]
        
        c_point = Point(center)
        
        # Check safety distance from start/end
        if start_point is not None and c_point.distance(Point(start_point)) < (safe_radius + r):
            continue
        if end_point is not None and c_point.distance(Point(end_point)) < (safe_radius + r):
            continue
            
        # Check overlap with existing circles
        new_circle = c_point.buffer(r)
        if any(Point(c[0], c[1]).buffer(c[2]).intersects(new_circle) for c in circles):
            continue
            
        circles.append((center[0], center[1], r))
        current_area += np.pi * (r**2)
        
    return circles


def generate_circles_fast(occupancy_rate, xmin, xmax, ymin, ymax, r_min, r_max,
                          start_point=None, end_point=None, safe_radius=0.0):
    workspace_area = (xmax - xmin) * (ymax - ymin)
    target_area = occupancy_rate * workspace_area
    current_area = 0.0
    circles = []
    centers = []
    tries = 0
    while current_area < target_area and tries < 5000:
        tries += 1
        r = random.uniform(r_min, r_max)
        cx = random.uniform(xmin + r, xmax - r)
        cy = random.uniform(ymin + r, ymax - r)
        if start_point is not None:
            dx = cx - start_point[0]
            dy = cy - start_point[1]
            if dx*dx + dy*dy < (safe_radius + r)**2:
                continue
        if end_point is not None:
            dx = cx - end_point[0]
            dy = cy - end_point[1]
            if dx*dx + dy*dy < (safe_radius + r)**2:
                continue
        overlap = False
        for (ox, oy, or_), (ecx, ecy) in zip(circles, centers):
            dx = cx - ecx
            dy = cy - ecy
            if dx*dx + dy*dy < (r + or_)**2:
                overlap = True
                break
        if overlap:
            continue
        circles.append((cx, cy, r))
        centers.append(np.array([cx, cy]))
        current_area += np.pi * (r**2)
    return circles


def random_convex_polygon(center, radius, n_vertices=6):
    """Generate a random convex polygon around a center."""

def generate_polygons(Npoly, xmin, xmax, ymin, ymax, poly_size, safe_zones, A_min, A_max):
    workspace = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    polygons = []
    tries = 0
    while len(polygons) < Npoly and tries < 10000:
        tries += 1
        center = [random.uniform(xmin+0.3, xmax-0.3),
                  random.uniform(ymin+0.3, ymax-0.3)]
        poly = random_convex_polygon(center, poly_size, n_vertices=random.randint(4,8))

        # check inside bounds
        if not workspace.contains(poly):
            continue

        # check intersection with safe zones
        if any(poly.intersects(sz) for sz in safe_zones):
            continue

        # check intersection with other polygons
        if any(poly.intersects(p) for p in polygons):
            continue

        # check area limits
        if poly.area < A_min or poly.area > A_max:
            continue

        polygons.append(poly)
    return polygons

def detailed_collision_with_polygons(curve_points, polygons):
    """Return number of colliding segments, length inside obstacles, and pieces for plotting."""    
    n_colliding_segments = 0
    inside_length = 0.0
    inside_segments = []
    outside_segments = []

    # Break curve into small segments
    for i in range(len(curve_points)-1):
        seg = LineString([curve_points[i], curve_points[i+1]])
        inside = False
        for poly in polygons:
            if seg.intersects(poly):
                n_colliding_segments += 1
                inter = seg.intersection(poly)
                inside_length += inter.length
                if not inter.is_empty:
                    inside_segments.append(inter)
                inside = True
        if not inside:
            outside_segments.append(seg)

    return n_colliding_segments, inside_length, inside_segments, outside_segments

##### Check functions

def check_inside_circle(point, pos_circle=[0.0, 0.0], r_circle=1):
        d = (pos_circle[0] - point[0])**2 + (pos_circle[1] - point[1])**2
        if  d <= r_circle**2:
            return True, d
        return False, d


def check_inside_ellipse(point, center=[0.0, 0.0], axes=[1.0, 1.0], theta=0.0):
    """
    Check if a point is inside an ellipse.

    Args:
        point: Coordinates of the point as [x, y].
        center: Center of the ellipse as [x_c, y_c].
        axes: Semi-major and semi-minor axes as [a, b].
        theta: Rotation angle of the ellipse in radians (default is 0).

    Returns:
        Tuple (bool, float), where bool indicates if the point is inside the ellipse,
        and float is the normalized distance (<=1 means inside).
    """
    x, y = point
    x_c, y_c = center
    a, b = axes
    
    # Translate point to ellipse center
    x_translated = x - x_c
    y_translated = y - y_c
    
    # Rotate point by -theta to align ellipse with axes
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rotated = cos_theta * x_translated + sin_theta * y_translated
    y_rotated = -sin_theta * x_translated + cos_theta * y_translated
    
    # Calculate normalized distance in ellipse equation
    d = (x_rotated / a)**2 + (y_rotated / b)**2
    
    if d <= 1:
        return True, d
    return False, d

def check_inside_sphere(point, pos_circle=[0.0, 0.0, 0.0], r_circle=1):
        
        if (pos_circle[0] - point[0])**2 + (pos_circle[1] - point[1])**2 + (pos_circle[2] - point[2])**2 <r_circle**2:
            return True
        return False


def check_inside_ellipsoid(point, pos_circle=[0.0, 0.0, 0.0], r_circle=[1.,1.,1.]):
        
        if ((pos_circle[0] - point[0])**2)/r_circle[0]**2 + ((pos_circle[1] - point[1])**2)/r_circle[1]**2 + \
           ((pos_circle[2] - point[2])**2)/r_circle[2]**2 <=1:
            return True
        return False

def check_in_line_ab(point, line_a=1, line_b=0):
    if (line_a*point[0] + line_b == point[1]):
        return True
    return False
    
def check_in_line(point, line_p=np.array([0,0]), line_v=np.array([0,1])):
    p = np.array([line_p[1]-point[1],-line_p[0]+point[0]])
    if line_v.dot(p) == 0:
        return True
    return False

def generate_segments(points, min_length=2, max_length=None):
    """Generate segments from a set of points."""
    if max_length is None:
        max_length = len(points) // 2
    
    segments = []
    for length in range(min_length, max_length + 1):
        for i in range(len(points) - length):
            segments.append(points[i:i+length])
    return segments

def check_segment_collision(segment, obstacles, r, tol=1):
    # Convert inputs to numpy arrays
    segment = np.array(segment)
    obstacles = [(np.array(ob[:2]), ob[2], ob[3]) for ob in obstacles]
    
    # Create BallTree for cylinder centers
    tree = BallTree(np.array([ob[0] for ob in obstacles]), leaf_size=40)
    
    # Perform approximate nearest neighbor search
    distances, indices = tree.query(segment[:, :2], k=1)
    
    # Convert indices to integers
    indices = indices.astype(int)
    
    # Check collisions
    collision_points = []
    for i, idx in enumerate(indices):
        center = obstacles[idx[0]][0]
        radius_x, radius_y = obstacles[idx[0]][1:]
        
        # Calculate distance between point and cylinder center
        dx = segment[i, 0] - center[0]
        dy = segment[i, 1] - center[1]
        dist_to_center = np.sqrt(dx**2 + dy**2)
        
        # Check if point is inside cylinder
        if dist_to_center <= r + radius_x and dist_to_center <= r + radius_y:
            collision_points.append(i)
    
    return np.array(collision_points)


# def check_collisions_cylinderBT(points_curve, cylinders, r=0.1):
#     centers = np.array([[c[0], c[1]] for c in cylinders])    
#     bt = BallTree(centers, leaf_size=40)
#     min_distances, indices = bt.query(points_curve[:,:2], k=1)    
#     dobs = []
#     for dist, ii in zip(min_distances, indices):     
#         idx = ii[0]        
#         if dist < cylinders[idx][2] + r:  # Check if point is inside cylinder            
#             dobs.append(dist)    
#     return dobs
def check_collisions_cylinderBT(points_curve, cylinders, r=0.1):
    """Return penetration depth for each point inside an expanded obstacle.

    *cylinders* must already have the robot radius included
    (i.e. *expanded_obstacles*). The *r* parameter is ignored and kept
    only for backward compatibility.
    """
    dobs = []
    pts = np.asarray(points_curve)
    for ob in cylinders:
        ox, oy, size = ob[0], ob[1], ob[2]
        d = np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)
        inside = d < size
        if np.any(inside):
            dobs.extend((size - d[inside]).tolist())
    return dobs

def check_collision_cylinder2line(curve, obs_cylinders, r, debug=0, tol=1):
    """ Check if a segment line collides with an infinite elliptical cylinder.
    curve: points [x,y,z] of a curve
    obs_cylinders: list of infinite elliptical cylinders [([x,y], radius_x, radius_y), ...]
    r: robot's radius    
    """
    curve_diff = np.diff(curve.T).T
    d = np.zeros(0)
    r = r/tol
    for ob in obs_cylinders:
        o = np.array([ob[0], ob[1], 0.0])
        li = curve[:-1] - o
        norm_pfi = np.linalg.norm(curve_diff, axis=1)
        
        vi = -np.sum(li * curve_diff, axis=1) / norm_pfi
        vi = vi[:, np.newaxis] * curve_diff

        pci = li + vi
        # closest point inside cylinder
        di = ((1 / (r + ob[2]))**2) * pci[:, 0]**2 + ((1 / (r + ob[3]))**2) * pci[:, 1]**2
        d = np.concatenate((d, di))

        # initial point inside cylinder
        di = ((1 / (r + ob[2]))**2) * (curve[:-1, 0] - ob[0])**2 + ((1 / (r + ob[3]))**2) * (curve[:-1, 1] - ob[1])**2
        d = np.concatenate((d, di))

        # end point inside cylinder
        di = ((1 / (r + ob[2]))**2) * (curve[:-1, 0] + curve_diff[:, 0] - ob[0])**2 + ((1 / (r + ob[3]))**2) * (curve[:-1, 1] + curve_diff[:, 1] - ob[1])**2
        d = np.concatenate((d, di))

    d = d[d**0.5 <= 1]

    return d


    
#### Random functions


def randomColor(n=1, use_list_colors=True):
    '''Random a color to use in matplotlib
    n = number of colors returned 
    '''
    if (use_list_colors) and (n <= 26):
         list_colors = ['blue', 'red', 'indigo', 'lime', 'sienna', 'magenta', 'darkkhaki', 'cyan', 'olive', 'chocolate', 'silver', 'orangered', 'yellow', 'orange', 'yellowgreen', 'teal', 'violet', 'blueviolet', 'cornflowerblue', 'darkkhaki', 'dimgray', 'tan', 'orchid', 'pink', 'crimson', 'chocolate', 'gold']
         colors = list_colors[0:n]
    else:
        colors = []
        
        for i in range(n):
            colors.append("#"+''.join(np.random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']) for i in range(6)))   
        
    if n==1: return colors[0]
    return colors


### Plots
### PLOTS

def plot_ellipsoid(center=(0,0,0), dimensions=(0.1,0.1,0.1), ax=None, Npoints=100, color='red', alpha=0.2):

    u = np.linspace(0.0, 2.0 * np.pi, Npoints)
    v = np.linspace(0.0, np.pi, Npoints)
    z = center[2] + dimensions[2]*np.outer(np.cos(u), np.sin(v))
    y = center[1] + dimensions[1]*np.outer(np.sin(u), np.sin(v))
    x = center[0] + dimensions[0]*np.outer(np.ones_like(u), np.cos(v))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=color, alpha=alpha)
    
    return ax

def draw_disc(p=np.array([0, 0]), r=1, ax=None, color='blue', fill=True, alpha=0.5, label="", zorder=None):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    if zorder is None:
        circle = plt.Circle((p[0], p[1]),radius=r, alpha=alpha, facecolor=color, edgecolor='none', label=label)    
    else:
        circle = plt.Circle((p[0], p[1]),radius=r, alpha=alpha, facecolor=color, edgecolor='none', label=label, zorder=zorder)    
    ax.add_artist(circle)
    return ax

def draw_disc3D(p=np.array([0, 0, 0]), r=1, zdir="z", ax=None, color='blue', fill=True, alpha=0.5):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
    p1 = Circle((p[0], p[1]), r, facecolor=color, edgecolor='none',alpha=alpha)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=p[2], zdir=zdir)
    return ax

def plot_cylinder(center=[0,0,0], rx=1, ry=1, height=1, min_height=0, ax=None, color_surface = 'green', alpha=0.5, color_wireframe='green', line_wireframe=0.5, orientation='z',zorder=1):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(min_height, height, 50)
    U, V = np.meshgrid(u, v)

    if orientation == 'x':
        x = V + center[2]
        y = ry * np.sin(U) + center[1]
        z = rx * np.cos(U) + center[0]
    elif orientation == 'y':
        x = rx * np.cos(U) + center[0]
        y = V + center[2]
        z = ry * np.sin(U) + center[1]
    elif orientation == 'z':
        x = rx * np.cos(U) + center[0]
        y = ry * np.sin(U) + center[1]
        z = V + center[2]
    # x = rx * np.cos(U) * orientation[0] + center[0]
    # y = ry * np.sin(U) * orientation[1] + center[1]
    # z = V + center[2]

    ax.plot_surface(x, y, z, alpha=alpha, color=color_surface, zorder=zorder)
    ax.plot_wireframe(x, y, z, color=color_wireframe, linewidth=line_wireframe, alpha=alpha/2,rstride=4, cstride=4, zorder=zorder)

    return ax

def plot_line3D(p_start=[0,0,0], p_end=[0,1,0], color='red', alpha=1):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], [p_start[2], p_end[2]], color=color, alpha=alpha)
    return ax

def plot_ellipse(center, rx, ry, angle=0, ax=None, Npoints=100, color='blue', alpha=1.0):
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
    t = np.linspace(0, 2*np.pi, Npoints)
    x = center[0] + (rx) * np.cos(t) * np.cos(angle) - (ry) * np.sin(t) * np.sin(angle)
    y = center[1] + (rx) * np.cos(t) * np.sin(angle) + (ry) * np.sin(t) * np.cos(angle)

    ax.fill(x, y, color=color, alpha=alpha)

#### String functions
def flatten_list(nested_list):
    """
    Função para transformar uma lista aninhada em uma lista achatada.
    """
    
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


