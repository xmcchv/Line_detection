#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import open3d as o3d
import threading

class KnotDetector:
    def __init__(self):
        rospy.init_node('knot_detector')
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None

        # 订阅相机内参
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        # 同步订阅RGB和深度图像
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.image_callback)

        # 初始化参数
        self.canny_low = 20
        self.canny_high = 70
        self.hough_threshold = 25

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.camera_info_sub.unregister()

    def image_callback(self, rgb_msg, depth_msg):
        if None in (self.fx, self.fy, self.cx, self.cy):
            rospy.logwarn("Camera info not received yet.")
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            rospy.logerr(e)
            return

        # 检测线段
        lines = self.detect_lines(rgb_image, depth_image)
        if lines is None:
            return

        valid_lines_3d = []
        for line in lines:
            x1, y1, x2, y2 = line
            d1 = self.get_depth_at_point(depth_image, x1, y1)
            d2 = self.get_depth_at_point(depth_image, x2, y2)
            if d1 is None or d2 is None:
                continue

            p1 = self.pixel_to_3d(x1, y1, d1)
            p2 = self.pixel_to_3d(x2, y2, d2)
            valid_lines_3d.append((p1, p2, (x1, y1, x2, y2)))

        intersecting_pairs = []
        for i in range(len(valid_lines_3d)):
            for j in range(i + 1, len(valid_lines_3d)):
                line1_2d = valid_lines_3d[i][2]
                line2_2d = valid_lines_3d[j][2]
                intersect = self.line_intersection(line1_2d, line2_2d)
                if intersect is not None:
                    intersecting_pairs.append((i, j, intersect))

        min_distance_threshold = 0.01  # 10厘米
        knot_pairs = []
        for pair in intersecting_pairs:
            i, j, point = pair
            line1 = valid_lines_3d[i]
            line2 = valid_lines_3d[j]
            distance = self.min_distance_between_lines(line1[0], line1[1], line2[0], line2[1])
            if distance < min_distance_threshold:
                knot_pairs.append((i, j))

        self.visualize(rgb_image, valid_lines_3d, intersecting_pairs, knot_pairs)
       
    def show_image(self, image, name):
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def visualize_process(self, steps):
        """显示处理过程各阶段图像"""
        titles = ["Original", "Enhanced", "Edges", "Merged Lines"]
        
        # 确保所有图像都是三维的
        processed_steps = []
        for img in steps:
            if len(img.shape) == 2:  # 如果是灰度图像
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为三通道
            processed_steps.append(img)
        
        # 进行图像堆叠
        stacked = np.hstack([cv2.resize(img, (400, 300)) for img in processed_steps])
        
        # 添加标题
        for i, title in enumerate(titles):
            cv2.putText(stacked, title, (10 + i * 400, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Processing Pipeline", stacked)
        cv2.waitKey(1)

    def generate_depth_mask(self, depth_image):
        """生成有效深度区域掩膜"""
        valid_depth = np.where((depth_image > 500) & (depth_image < 3000), 255, 0)
        valid_depth = valid_depth.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        smoothed = cv2.morphologyEx(valid_depth, cv2.MORPH_CLOSE, kernel)
        return smoothed

    def auto_canny_edge_detection(self, image, sigma=0.33):
        md = np.median(image)
        lower_value = int(max(0, (1.0 - sigma) * md))
        upper_value = int(min(255, (1.0 + sigma) * md))
        return cv2.Canny(image, lower_value, upper_value)

    def detect_lines(self, image, depth_image):
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=300,
            maxLineGap=10
        )
        
        print(f'lines: {len(lines)}', end='')
        if lines is not None:
            lines = self.merge_lines(lines, angle_thresh=45, dist_thresh=30, img_width=width, img_height=height, edge_buffer=50)
        print(f' after lines: {len(lines)}')

        line_image = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        steps = [image, binary, line_image]
        self.visualize_process(steps)
        # 提取每个 line[0] 的坐标到一个新的列表
        extracted_lines = []
        if lines is not None:
            for line in lines:
                # 假设 line[0] 是一个包含 [x1, y1, x2, y2] 的列表
                if len(line) > 0 and len(line[0]) == 4:
                    extracted_lines.append(line[0])  # 提取坐标
        # 将提取的线段转换为 NumPy 数组
        if len(extracted_lines) > 0:
            lines_array = np.array(extracted_lines)  # 转换为 NumPy 数组
            return lines_array.reshape(-1, 4)  # 确保返回形状为 (-1, 4)
        else:
            return np.array([]).reshape(0, 4)  # 返回一个空的 NumPy 数组，形状为 (0, 4)

    def merge_lines(self, lines, angle_thresh=10, dist_thresh=20, img_width=640, img_height=480, edge_buffer=20):
        final_lines = []

        if len(lines) == 0 or any(len(line[0]) != 4 for line in lines):
            return np.array(final_lines)

        # 过滤靠近图像边缘的线段
        lines = [
            line for line in lines
            if (line[0][0] > edge_buffer and line[0][1] > edge_buffer and
                line[0][2] < img_width - edge_buffer and line[0][3] < img_height - edge_buffer)
        ]
        print(f' 过滤图像边缘:{len(lines)}', end='')

        # 过滤非接近垂直的线段
        lines = [
            line for line in lines
            if self.is_near_vertical(line[0], angle_thresh)
        ]
        print(f' 过滤非垂直:{len(lines)}', end='')
        return lines

    def is_near_vertical(self, line, angle_thresh):
        # 计算线段的角度
        theta = np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi
        # 判断是否接近垂直
        return (90 - angle_thresh) <= abs(theta) <= (90 + angle_thresh)


    def get_depth_at_point(self, depth_image, x, y):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            depth = depth_image[y, x]
            return depth / 1000.0 if depth != 0 else None
        return None

    def pixel_to_3d(self, u, v, depth):
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        return np.array([X, Y, depth])

    def line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        return None

    def min_distance_between_lines(self, a0, a1, b0, b1):
        def np_array(point):
            return np.array(point) if not isinstance(point, np.ndarray) else point

        a0, a1, b0, b1 = map(np_array, [a0, a1, b0, b1])
        A = a1 - a0
        B = b1 - b0
        cross = np.cross(A, B)
        denom = np.linalg.norm(cross) ** 2

        if denom < 1e-6:
            return min(
                np.linalg.norm(a0 - b0),
                np.linalg.norm(a0 - b1),
                np.linalg.norm(a1 - b0),
                np.linalg.norm(a1 - b1)
            )
        else:
            t = (b0 - a0)
            mat = np.array([
                [np.dot(A, A), -np.dot(A, B)],
                [np.dot(A, B), -np.dot(B, B)]
            ])
            vec = np.array([np.dot(A, t), np.dot(B, t)])
            t_u, t_v = np.linalg.solve(mat, vec)
            t_u = max(0, min(1, t_u))
            t_v = max(0, min(1, t_v))
            closest_a = a0 + t_u * A
            closest_b = b0 + t_v * B
            return np.linalg.norm(closest_a - closest_b)

    def visualize(self, image, lines_3d, intersecting_pairs, knot_pairs):
        # 绘制所有线段
        for line in lines_3d:
            x1, y1, x2, y2 = line[2]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 绘制相交点
        for pair in intersecting_pairs:
            x, y = pair[2]
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        # 绘制打结的线段对
        for i, j in knot_pairs:
            line1 = lines_3d[i][2]
            line2 = lines_3d[j][2]
            cv2.line(image, (int(line1[0]), int(line1[1])), (int(line1[2]), int(line1[3])), (255, 0, 0), 3)
            cv2.line(image, (int(line2[0]), int(line2[1])), (int(line2[2]), int(line2[3])), (255, 0, 0), 3)

        cv2.imshow("Knot Detection", image)
        cv2.waitKey(1)

if __name__ == '__main__':
    detector = KnotDetector()
    rospy.spin()
