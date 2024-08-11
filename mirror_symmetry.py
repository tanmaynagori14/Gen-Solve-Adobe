import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

# SIFT and BFMatcher initialization
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

class MirrorSymmetryDetection:
    def __init__(self, image_path: str):
        self.image = self._read_color_image(image_path)
        self.reflected_image = np.fliplr(self.image)  # Flipped version of the image
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.reflected_image, None)

    def _read_color_image(self, image_path):
        image = cv2.imread(image_path)
        b, g, r = cv2.split(image)
        return cv2.merge([r, g, b])

    def find_matchpoints(self):
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        matchpoints = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        return sorted(matchpoints, key=lambda x: x.distance)

    def find_points_r_theta(self, matchpoints):
        points_r = []
        points_theta = []
        for match in matchpoints:
            point = self.kp1[match.queryIdx]
            mirpoint = self.kp2[match.trainIdx]

            mirpoint.angle = np.deg2rad(mirpoint.angle)
            mirpoint.angle = np.pi - mirpoint.angle
            if mirpoint.angle < 0.0:
                mirpoint.angle += 2 * np.pi

            mirpoint.pt = (self.reflected_image.shape[1] - mirpoint.pt[0], mirpoint.pt[1])
            theta = angle_with_x_axis(point.pt, mirpoint.pt)
            xc, yc = midpoint(point.pt, mirpoint.pt)
            r = xc * np.cos(theta) + yc * np.sin(theta)

            points_r.append(r)
            points_theta.append(theta)

        return points_r, points_theta

    def draw_matches(self, matchpoints, top=10):
        img = cv2.drawMatches(self.image, self.kp1, self.reflected_image, self.kp2,
                               matchpoints[:top], None, flags=2)
        plt.imshow(img)
        plt.title("Top {} pairs of symmetry points".format(top))
        plt.savefig('static/matchpoints.png')

    def draw_hex(self, points_r, points_theta):
        image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
        plt.colorbar()
        plt.savefig('static/hexbin.png')

    def find_coordinate_maxhexbin(self, image_hexbin, sorted_vote, vertical):
        for k, v in sorted_vote.items():
            if vertical:
                return k[0], k[1]
            else:
                if k[1] == 0 or k[1] == np.pi:
                    continue
                else:
                    return k[0], k[1]

    def sort_hexbin_by_votes(self, image_hexbin):
        counts = image_hexbin.get_array()
        ncnts = np.count_nonzero(np.power(10, counts))
        verts = image_hexbin.get_offsets()
        output = {}

        for offc in range(verts.shape[0]):
            binx, biny = verts[offc][0], verts[offc][1]
            if counts[offc]:
                output[(binx, biny)] = counts[offc]
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

    def draw_mirror_line(self, r, theta, title):
        line_image = self.image.copy()
        height, width = line_image.shape[:2]
        for y in range(height):
            try:
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                if 0 <= x < width:
                    line_image[y, x] = [0, 0, 255]
                    if x + 1 < width:
                        line_image[y, x + 1] = [0, 0, 255]
            except IndexError:
                continue

        plt.imshow(line_image)
        plt.axis('off')
        plt.title(title)
        plt.savefig(f'static/{title}.png')

def angle_with_x_axis(pi, pj):
    x, y = pi[0] - pj[0], pi[1] - pj[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle

def midpoint(pi, pj):
    return (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2

def detect_mirror_line(image_path):
    mirror = MirrorSymmetryDetection(image_path)
    matchpoints = mirror.find_matchpoints()
    points_r, points_theta = mirror.find_points_r_theta(matchpoints)
    mirror.draw_matches(matchpoints, top=10)
    mirror.draw_hex(points_r, points_theta)
    
    image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
    sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
    r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)
    
    mirror.draw_mirror_line(r, theta, "Detected Mirror Line")

# if __name__ == "__main__":
#     detect_mirror_line('butterfly.png')
