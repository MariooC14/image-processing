# python
import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt

# 3-D, unit-cube (homogenous coordinates)
P = np.array([
    [0,0,0,1], [1,0,0,1], [1,1,0,1], [0,1,0,1],
    [0,0,1,1], [1,0,1,1], [1,1,1,1], [0,1,1,1]
])

# cube edges (pairs of vertex indices)
edges = [
    (0,1),(1,2),(2,3),(3,0),  # bottom
    (4,5),(5,6),(6,7),(7,4),  # top
    (0,4),(1,5),(2,6),(3,7)   # verticals
]

deltaRy = 5  # degrees per frame
iters = 0

# camera / projection params
f = 1
Tx = 0.0
Ty = 0.0
Tz = 2.0  # move camera back enough to see cube

K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])

# prepare interactive plot
plt.ion()
fig, ax = plt.subplots()
scatter_plot, = ax.plot([], [], 'bo')             # vertices
edges_plot, = ax.plot([], [], 'k-')               # edges as segmented line with NaNs

ax.set_title('3D Cube Projection (rotating)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal', 'box')

# set fixed plot limits (centered around 0.5 since cube is [0,1]^3)
lim = 0.5
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

while True:
    # rotations in radians
    Rx = np.radians(30)
    Ry = np.radians(iters * deltaRy)
    Rz = np.radians(0)

    RMx = np.array([
        [1, 0, 0],
        [0, cos(Rx), -sin(Rx)],
        [0, sin(Rx), cos(Rx)]
    ])
    RMy = np.array([
        [cos(Ry), 0, sin(Ry)],
        [0, 1, 0],
        [-sin(Ry), 0, cos(Ry)]
    ])
    RMz = np.array([
        [cos(Rz), -sin(Rz), 0],
        [sin(Rz), cos(Rz), 0],
        [0, 0, 1]
    ])
    RM = RMz @ RMy @ RMx

    M = np.zeros((3,4))
    M[0:3, 0:3] = RM
    M[0,3] = Tx
    M[1,3] = Ty
    M[2,3] = Tz

    # perspective projection
    p = K @ M @ P.T
    x = p[0, :] / p[2, :]
    y = p[1, :] / p[2, :]

    # update scatter (vertices)
    scatter_plot.set_data(x, y)

    # build segmented edge line (use NaN separators)
    xs = []
    ys = []
    for i, j in edges:
        xs.extend([x[i], x[j], np.nan])
        ys.extend([y[i], y[j], np.nan])
    edges_plot.set_data(xs, ys)

    # redraw
    fig.canvas.draw_idle()
    plt.pause(0.03)  # pause ~30ms

    iters += 1
