import math

from numpy.ma.core import argmin


def get_orientation_angle_radians(x1, y1, x2, y2, x3, y3, x4, y4):
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4

    orientation_x, orientation_y = get_orientation_arrow_point(x1, y1, x2, y2, x3, y3, x4, y4)

    angle = math.atan2(orientation_y - center_y, center_x - orientation_x)
    return angle

def get_orientation_arrow_point(x1, y1, x2, y2, x3, y3, x4, y4):
    orientation_x = (x2 + x3) / 2
    orientation_y = (y2 + y3) / 2
    return orientation_x, orientation_y

def rotate_points(x1, y1, x2, y2, x3, y3, x4, y4, angle):
    # if rotation > math.pi / 2:
    #     x4, y4, x1, y1, x2, y2, x3, y3 = torch.flatten(xyxyxyxyn).tolist()
    # else:

    # # Mirroring calculation here https://docs.ultralytics.com/datasets/obb/#yolo-obb-format
    # l1 = x1 - x4
    # l2 = y4 - y1

    # if l1 > l2:
    #     x4, y4, x1, y1, x2, y2, x3, y3 = torch.flatten(xyxyxyxyn).tolist()

    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    point_with_min_x = min(points, key=lambda point: point[0])
    point_with_min_y = min(points, key=lambda point: point[1])

    l1 = point_with_min_y[0] - point_with_min_x[0]
    l2 = point_with_min_x[1] - point_with_min_y[1]

    if l1 > l2:
        top_left_index = points.index(point_with_min_x)
    else:
        top_left_index = points.index(point_with_min_y)

    points_to_shift = top_left_index
    while points_to_shift > 0:
        point = points.pop(0)
        points.append(point)
        points_to_shift -= 1

    (xa, ya), (xb, yb), (xc, yc), (xd, yd) = points

    return xa, ya, xb, yb, xc, yc, xd, yd