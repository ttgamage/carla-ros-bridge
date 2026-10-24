#!/usr/bin/env python

"""

Layout: the fields of ouster_ros::Point, read by FAST-LIO

    x, y, z, intensity  float32   right-handed (x forward, y left, z up), m; intensity (0..1)
    t                   uint32    ns from the header stamp (the first slice of the turn)
    reflectivity        uint16    intensity * 255
    ring                uint8     CARLA channel, 0 = top beam (upper_fov), as in Ouster
    ambient             uint16    0 (no ambient image in CARLA)
    range               uint32    mm

Slices: at every simulation step CARLA's ray-cast LiDAR returns only the angle swept during that step
(rotation_frequency * 360 * fixed_delta_seconds deg), computed with the sensor at its pose at the end of the
step and stamped with that time. TurnAccumulator joins ticks_per_turn = 1 / (rotation_frequency * step)
consecutive slices into one turn, counting simulation frames: consecutive slices
are adjacent sectors, so any ticks_per_turn consecutive ones cover 360 deg, wherever they start.
Each slice keeps the frame of its own instant, as with a real spinning LiDAR, and gets
t = its stamp - the turn's first stamp, so the consumer can deskew the turn with the IMU.
"""

import numpy as np

OUSTER_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'ambient', 'range'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<u4', '<u2', 'u1', '<u2', '<u4'],
    'offsets': [0, 4, 8, 12, 16, 20, 22, 24, 28],
    'itemsize': 32,
})

# sensor_msgs/PointField datatypes
_POINTFIELD_DATATYPE = {np.dtype('<f4'): 7, np.dtype('<u4'): 6, np.dtype('<u2'): 4, np.dtype('u1'): 2}

TIME_MODES = ('sweep', 'zero')

# 1 / (rotation_frequency * step) farther than this from an integer: turns are not whole numbers of slices
_STEPS_TOL = 1e-3


def ouster_fields():
    return [(name, OUSTER_DTYPE.fields[name][1], _POINTFIELD_DATATYPE[OUSTER_DTYPE.fields[name][0]])
            for name in OUSTER_DTYPE.names]


def slice_points(raw_data, channel_counts):
    """
    One CARLA LidarMeasurement -> (xyz Nx3 float32 right-handed, intensity N float32, ring N uint8).

    raw_data: x y z intensity 
    """
    raw = np.frombuffer(bytes(raw_data), dtype=np.float32).reshape(-1, 4)
    counts = np.asarray(channel_counts, dtype=np.int64)
    if counts.sum() != len(raw):
        raise ValueError("channel counts sum to {}, the slice has {} points".format(counts.sum(), len(raw)))
    if len(counts) > 256:
        raise ValueError("ring is uint8: at most 256 channels")
    ring = np.repeat(np.arange(len(counts), dtype=np.uint8), counts)
    xyz = raw[:, :3].copy()
    xyz[:, 1] *= -1.0  # left-handed -> right-handed
    return xyz, raw[:, 3].copy(), ring


def pack_turn(slices, time_mode='sweep'):
    """
    slices: [(timestamp s, xyz, intensity, ring)] of one turn, in time order.
    Returns (stamp s, numpy array with OUSTER_DTYPE): stamp = first slice, points in time order.
    """
    stamp = slices[0][0]
    n = sum(len(s[1]) for s in slices)
    cloud = np.zeros(n, OUSTER_DTYPE)
    i = 0
    for timestamp, xyz, intensity, ring in slices:
        k = len(xyz)
        part = cloud[i:i + k]
        part['x'], part['y'], part['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        part['intensity'] = intensity
        part['ring'] = ring
        part['reflectivity'] = np.clip(np.rint(intensity * 255.0), 0, 65535).astype(np.uint16)
        part['range'] = np.rint(np.linalg.norm(xyz.astype(np.float64), axis=1) * 1000.0).astype(np.uint32)
        if time_mode == 'sweep':
            part['t'] = int(round((timestamp - stamp) * 1e9))
        i += k
    return stamp, cloud


class TurnAccumulator(object):

    """
    Joins the per-step slices of a ray-cast LiDAR into full turns, counting simulation frames.

    The step is measured from the data: between two slices of consecutive frames,
    step = their timestamp difference, and ticks_per_turn = round(1 / (rotation_frequency * step)).
    Turns are ticks_per_turn frames long, counted from the frame where the step was measured
    (the origin): turn n holds frames origin + n * ticks_per_turn ... + ticks_per_turn - 1.
    - A turn is published at its last frame, with the slices that arrived (one lost in the middle: the
      turn has a gap; the last one lost: the partial turn is dropped when the next turn starts).
    - If the step changes (the server switched between asynchronous and synchronous mode, or to another
      fixed_delta_seconds), ticks_per_turn is recomputed and counting restarts from that frame.
    With rotation_frequency = 1 / step every slice is a whole instantaneous turn (all t = 0).
    exact is False when 1 / (rotation_frequency * step) is not an integer: turns then cover a bit more or
    less than 360 deg.

    time_mode: 'sweep' sets t per slice; 'zero' sets every t to 0 (same points, stamp at the first slice).
    """

    def __init__(self, rotation_frequency, time_mode='sweep'):
        if rotation_frequency <= 0:
            raise ValueError("rotation_frequency must be > 0")
        if time_mode not in TIME_MODES:
            raise ValueError("time_mode must be one of {}".format(TIME_MODES))
        self.frequency = float(rotation_frequency)
        self.time_mode = time_mode
        self.ticks_per_turn = None
        self.exact = True
        self._origin = None
        self._prev = None      # (frame, timestamp) of the previous slice
        self._turn = None
        self._slices = []

    def _measure_step(self, frame, timestamp):
        """Returns True if ticks_per_turn was (re)computed from this slice and the previous one."""
        prev, self._prev = self._prev, (frame, timestamp)
        if prev is None or frame - prev[0] != 1 or timestamp <= prev[1]:
            return False
        steps = 1.0 / (self.frequency * (timestamp - prev[1]))
        ticks = max(1, int(round(steps)))
        if ticks == self.ticks_per_turn:
            return False
        self.ticks_per_turn = ticks
        self.exact = abs(steps - ticks) < _STEPS_TOL * ticks
        self._origin = frame
        self._turn = None
        self._slices = []
        return True

    def add(self, frame, timestamp, xyz, intensity, ring):
        """Adds the slice of a simulation frame. Returns (stamp, cloud) when a turn is complete, else None."""
        self._measure_step(frame, timestamp)
        if self.ticks_per_turn is None or frame < self._origin:
            return None
        turn, index = divmod(frame - self._origin, self.ticks_per_turn)
        if turn != self._turn:
            self._slices = []  # leftovers of a turn whose last slice was lost
            self._turn = turn
        self._slices.append((timestamp, xyz, intensity, ring))
        if index != self.ticks_per_turn - 1:
            return None
        result = pack_turn(self._slices, self.time_mode)
        self._slices = []
        return result
