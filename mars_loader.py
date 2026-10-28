import numpy as np

class MarsTerrain:

    def __init__(self, filepath, width_deg=90, height_deg=44, ppd=128):
        self.filepath = filepath
        self.ppd = ppd

        self.rows = height_deg * ppd
        self.cols = width_deg * ppd

        # Memory-map file (int16 signed)
        self.data = np.memmap(
            filepath,
            dtype='>i2',   # Big-endian signed 16-bit
            mode='r',
            shape=(self.rows, self.cols)
        )

        # Mars mean radius baseline shift (optional normalization)
        self.elevation = np.array(self.data, dtype=np.float32)

    def extract_profile(self, lat_index=None):

        if lat_index is None:
            lat_index = self.rows // 2  # middle latitude

        profile = self.elevation[lat_index, :]

        # Normalize so lowest point is zero
        profile = profile - np.min(profile)

        return profile