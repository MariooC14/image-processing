"""
DLT Computation Server
Provides HTTP endpoint for computing camera matrix P from 3D-2D correspondences
using normalized DLT with proper SVD.
"""
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import sys

# Try to import OpenCV for better decomposition
try:
    import cv2
    HAS_CV2 = True
except:
    HAS_CV2 = False
    print("Warning: OpenCV not available, using manual RQ decomposition")

def normalize_2d(pts):
    """Normalize 2D points: translate to centroid, scale so mean distance = sqrt(2)"""
    pts = np.array(pts, dtype=float)
    n = len(pts)
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted**2).sum(axis=1)).mean()
    if mean_dist < 1e-10:
        mean_dist = 1.0
    scale = np.sqrt(2) / mean_dist

    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    normalized = shifted * scale
    return T, normalized

def normalize_3d(pts):
    """Normalize 3D points: translate to centroid, scale so mean distance = sqrt(3)"""
    pts = np.array(pts, dtype=float)
    n = len(pts)
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted**2).sum(axis=1)).mean()
    if mean_dist < 1e-10:
        mean_dist = 1.0
    scale = np.sqrt(3) / mean_dist
    
    U = np.array([
        [scale, 0,          0,  -scale * centroid[0]],
        [0,     scale,      0,  -scale * centroid[1]],
        [0,     0,      scale,  -scale * centroid[2]],
        [0,     0,          0,                     1]
    ])
    
    normalized = shifted * scale
    return U, normalized

def build_A(world_pts, image_pts):
    """Build the A matrix for DLT: A p = 0 (matching MATLAB row order)"""
    A = []
    for i in range(len(world_pts)):
        X, Y, Z = world_pts[i]
        u, v = image_pts[i]
        # MATLAB order: row1 = [0,0,0,0, -X,-Y,-Z,-1, v*X,v*Y,v*Z,v]
        #               row2 = [X,Y,Z,1,  0,0,0,0, -u*X,-u*Y,-u*Z,-u]
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
    return np.array(A)

def compute_dlt(world_pts, image_pts):
    """
    Compute camera projection matrix P using DLT (no normalization, matching MATLAB).
    Returns P (3x4), RMSE, rank of A.
    """
    print(f"\n=== DLT Computation (No Normalization - MATLAB Style) ===")
    print(f"Number of correspondences: {len(world_pts)}")
    print(f"Image points range: u=[{image_pts[:, 0].min():.1f}, {image_pts[:, 0].max():.1f}], v=[{image_pts[:, 1].min():.1f}, {image_pts[:, 1].max():.1f}]")
    print(f"World points range: X=[{world_pts[:, 0].min():.1f}, {world_pts[:, 0].max():.1f}], Y=[{world_pts[:, 1].min():.1f}, {world_pts[:, 1].max():.1f}], Z=[{world_pts[:, 2].min():.1f}, {world_pts[:, 2].max():.1f}]")

    # === NORMALIZATION ADDED BY [Van-Mario Caval] ===   #
    T, image_pts_norm = normalize_2d(image_pts)
    U, world_pts_norm = normalize_3d(world_pts)

    # Build A matrix directly from raw points (no normalization, like MATLAB)
    A = build_A(world_pts_norm, image_pts_norm)
    
    # SVD: A = U S V^T, solution is last column of V
    U_svd, S, Vt = np.linalg.svd(A, full_matrices=True)
    p = Vt[-1, :]  # Last row of V^T = last column of V
    
    # Reshape to 3x4 (standard C-order reshape, same as Qt app and common practice)
    P_tilde = p.reshape(3, 4)

    P = np.linalg.inv(T) @ P_tilde @ U
    # === END MODIFICATION ===

    # Do NOT flip/scale P's sign. Keeping P as returned by SVD preserves the
    # orientation of R and avoids unintended camera facing flips. P is up to scale
    # anyway, and reprojections are already good.
    print(f"\nP matrix (from SVD):\n{P}")

    
    # Compute reprojection error
    world_hom = np.hstack([world_pts, np.ones((len(world_pts), 1))])
    proj = (P @ world_hom.T).T  # Nx3
    proj_2d = proj[:, :2] / proj[:, 2:3]
    errors = image_pts - proj_2d
    rmse = np.sqrt((errors**2).sum(axis=1).mean())
    
    print(f"\nReprojection RMSE: {rmse:.3f} pixels")
    
    # Rank of A (number of singular values > threshold)
    rank = np.sum(S > 1e-8)
    
    return P, rmse, rank, S


def matlab_qr(A):
    """
    NumPy QR consistent with MATLAB/Octave:
      - full QR
      - positive diagonal in R
    """
    Q, R = np.linalg.qr(A, mode='complete')

    # Force diagonal(R) > 0 via a diagonal sign matrix S
    s = np.sign(np.diag(R))
    s[s == 0] = 1
    S = np.diag(s)

    # If we scale columns of Q by S, we must scale rows of R by S on the left
    Q = Q @ S
    R = S @ R
    return Q, R


def decompose_p(P):
    """Decompose P = K [R | t] using qr(inv(M)) with correct mapping and scaling."""

    print("P matrix before decomposition:")
    print(P)

    # Camera center (homogeneous nullspace of P)
    U_p, S_p, Vt_p = np.linalg.svd(P)
    camera_center_h = Vt_p[-1]
    C_null = camera_center_h[:3] / camera_center_h[3]
    print(f"Camera center from null space: {C_null}")

    # Split P = [M | p4]
    M  = P[:, :3]
    p4 = P[:, 3]

    # --- Core idea: inv(M) = Q R  =>  M = inv(R) inv(Q) = (R^{-1}) (Q^T)
    M_inv = np.linalg.inv(M)
    Q, Rq = matlab_qr(M_inv)     # diag(Rq) > 0 by our convention

    K0 = np.linalg.inv(Rq)       # upper-triangular (candidate K)
    R0 = Q.T                     # orthogonal (candidate R)

    # --- Make diag(K) > 0 without changing M = K R
    s = np.sign(np.diag(K0))
    s[s == 0] = 1
    S = np.diag(s)
    K = K0 @ S
    R = S @ R0                   # left-multiply flips corresponding rows in R

    # --- Ensure R is a proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        S2 = np.diag([1, 1, -1])
        K = K @ S2
        R = S2 @ R

    # --- Normalize K BEFORE solving for t (set K[2,2] = 1)
    k_scale = K[2, 2]
    if abs(k_scale) < 1e-15:
        raise ValueError("K[2,2] is zero; cannot normalize.")
    K  = K / k_scale
    p4 = p4 / k_scale            # keep p4 = K t consistent with the rescale

    # --- Solve for t from p4 = K t (avoid explicit inverse)
    t = np.linalg.solve(K, p4)

    # --- Camera center from extrinsics: C = -R^T t
    C = -R.T @ t

    # --- Optional: ensure camera faces a chosen scene point
    scene_center = np.array([1.5, 1.5, 1.5])
    forward_world = -R[:, 2]           # optical axis in world coords
    if np.dot(forward_world, scene_center - C) < 0:
        R = -R
        t = -t
        C = -R.T @ t
        forward_world = -R[:, 2]

    print(f"K (intrinsic matrix):\n{K}")
    print(f"R (rotation matrix):\n{R}")
    print(f"t (translation vector): {t}")
    print(f"Principal point: ({K[0,2]}, {K[1,2]})")
    print(f"Focal lengths: fx={K[0,0]}, fy={K[1,1]}")
    print(f"Camera center from -R^T*t (used): {C}")
    print(f"Camera center from null space (reference): {C_null}")
    print(f"Difference: {np.linalg.norm(C - C_null)}")
    print(f"Camera forward (world): {forward_world}")

    return K, R, t, C


class DLTHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle DLT computation request"""
        if self.path != '/compute_dlt':
            self.send_error(404)
            return
        
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body)
            world_pts = np.array(data['world_points'])
            image_pts = np.array(data['image_points'])
            
            if len(world_pts) < 6 or len(image_pts) < 6:
                raise ValueError("Need at least 6 correspondences")
            
            if len(world_pts) != len(image_pts):
                raise ValueError("Number of 3D and 2D points must match")
            
            # Compute P
            P, rmse, rank, singular_values = compute_dlt(world_pts, image_pts)
            
            # Decompose
            K, R, t, C = decompose_p(P)
            
            # Prepare response
            response = {
                'success': True,
                'P': P.tolist(),
                'K': K.tolist(),
                'R': R.tolist(),
                't': t.tolist(),
                'C': C.tolist(),
                'rmse': float(rmse),
                'rank': int(rank),
                'singular_values': singular_values.tolist()
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e)
            }
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Override to show cleaner logs"""
        print(f"[DLT Server] {format % args}")

def run_server(port=8765):
    """Start the DLT computation server"""
    server = HTTPServer(('localhost', port), DLTHandler)
    print(f"DLT Server running on http://localhost:{port}")
    print(f"Endpoint: POST http://localhost:{port}/compute_dlt")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

if __name__ == '__main__':
    port = 8765
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
