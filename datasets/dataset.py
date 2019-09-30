import copy
import chainer
import chainercv
import numpy as np

class KeypointDataset(chainer.dataset.DatasetMixin):
    """
    A simple dataset that for imgs and keypoints. Landmarks are returned as heatmaps or keypoints.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    def __init__(
        self, img_paths, pts_paths, mode
    ):
        self.img_paths = img_paths
        self.pts_paths = pts_paths
        self.mode = mode

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        img_path = self.img_paths[i]
        img = chainercv.utils.read_image(img_path)

        pts_path = self.pts_paths[i]
        pts = np.loadtxt(pts_path, skiprows=3, comments="}") # bit of a hack to read .pts file

        if self.mode == 'heatmap':
            heatmaps = self.generate_hm(img.shape[2], img.shape[1], pts).T
            return (img, heatmaps)
        else:
            return (img, pts)

    def gaussian_k(self, x0,y0,sigma, height, width):
        """
        Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        src:https://fairyonice.github.io
        """
        x = np.arange(0, width, 1, float) ## (width,)
        y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

    def generate_hm(self, height, width ,landmarks,s=3):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        src: https://fairyonice.github.io
        """
        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
        for i in range(Nlandmarks):
            if not np.array_equal(landmarks[i], [-1,-1]):

                hm[:,:,i] = self.gaussian_k(landmarks[i][0],
                                        landmarks[i][1],
                                        s,height, width)
            else:
                hm[:,:,i] = np.zeros((height,width))
        return hm
