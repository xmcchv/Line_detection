#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

class KnotDetector:
    def __init__(self):
        rospy.init_node('knot_detector')
        self.bridge = CvBridge()
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # 订阅相机内参
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        # 同步订阅RGB和深度图像
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.image_callback)

        # 动态参数注册
        self.canny_low = rospy.get_param("~canny_low", 25)
        self.canny_high = rospy.get_param("~canny_high", 70)
        self.hough_threshold = rospy.get_param("~hough_threshold", 20)
        
        # 参数变化回调
        rospy.Timer(rospy.Duration(1), self.param_update_cb)

    def param_update_cb(self, event):
        self.canny_low = rospy.get_param("~canny_low", 25)
        self.canny_high = rospy.get_param("~canny_high", 70)
        self.hough_threshold = rospy.get_param("~hough_threshold", 20)

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
           # 处理图像（例如显示或保存）
            # cv2.imshow("RGB Image", rgb_image)
            # cv2.imshow("Depth Image", depth_image)
            # cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(e)
            return

        # self.show_image(depth_image, "depth_image")
        # 检测线段
        lines = self.detect_lines(rgb_image)
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
        """
        生成有效深度区域掩膜：
        1. 排除无效深度区域
        2. 聚焦于合理工作距离
        """
        valid_depth = np.where((depth_image > 500) & (depth_image < 3000), 255, 0)
        valid_depth = valid_depth.astype(np.uint8)
        
        # 形态学处理填充空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        smoothed = cv2.morphologyEx(valid_depth, cv2.MORPH_CLOSE, kernel)
        
        return smoothed

    def detect_lines(self, image):
        # 步骤1: 光照归一化
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_eq = clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # 步骤2: 定向滤波（保留横向/纵向特征）
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        kernel_h = np.array([[ -1, 2, -1],
                            [ -1, 2, -1],
                            [ -1, 2, -1]], dtype=np.float32)
        kernel_v = kernel_h.T
        filtered_h = cv2.filter2D(gray, -1, kernel_h)
        filtered_v = cv2.filter2D(gray, -1, kernel_v)
        directional = cv2.addWeighted(filtered_h, 0.5, filtered_v, 0.5, 0)

        # 步骤3: 自适应对比度提升
        adaptive = cv2.adaptiveThreshold(
            directional, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 步骤4: 形态学重建
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

        # 步骤5: 改进的Canny检测
        blurred = cv2.GaussianBlur(morph, (5,5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high, L2gradient=True)
        # # 深度掩膜过滤（可选）
        # depth_mask = self.generate_depth_mask(depth_image)
        # edges = cv2.bitwise_and(edges, edges, mask=depth_mask)

        # 步骤6: 概率霍夫变换参数优化
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,          # 更低阈值以检测更多线段
            minLineLength=15,      # 允许更短线段
            maxLineGap=10          # 更大的间隙容忍度
        )

        # 线段合并算法（解决碎片化）
        if lines is not None:
            lines = self.merge_lines(lines, angle_thresh=10, dist_thresh=20)
        # 绘制检测到的线段
        line_image = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绘制线段
        
        # 在detect_lines中调用：
        steps = [image, enhanced, edges, line_image]
        self.visualize_process(steps)

        return lines.reshape(-1, 4) if lines is not None else []

    def merge_lines(self, lines, angle_thresh=10, dist_thresh=20):
        final_lines = []
        # 确保 lines 不是空的且每个线段都是有效的
        if len(lines) == 0 or any(len(line) != 4 for line in lines):
            return np.array(final_lines)

        lines = sorted(lines, key=lambda x: np.arctan2(x[3]-x[1], x[2]-x[0]))

        current_line = lines[0]
        for line in lines[1:]:
            theta1 = np.arctan2(current_line[3]-current_line[1], current_line[2]-current_line[0])
            theta2 = np.arctan2(line[3]-line[1], line[2]-line[0])
            angle_diff = np.abs(theta1 - theta2) * 180 / np.pi
            dist = np.linalg.norm(np.array(current_line[:2]) - np.array(line[2:]))

            if angle_diff < angle_thresh and dist < dist_thresh:
                x1 = min(current_line[0], line[0], current_line[2], line[2])
                y1 = min(current_line[1], line[1], current_line[3], line[3])
                x2 = max(current_line[0], line[0], current_line[2], line[2])
                y2 = max(current_line[1], line[1], line[3])
                current_line = [x1, y1, x2, y2]
            else:
                final_lines.append(current_line)
                current_line = line

        final_lines.append(current_line)
        return np.array(final_lines)


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
        # 计算两条线段之间的最短距离
        def np_array(point):
            return np.array(point) if not isinstance(point, np.ndarray) else point

        a0, a1, b0, b1 = map(np_array, [a0, a1, b0, b1])
        A = a1 - a0
        B = b1 - b0
        cross = np.cross(A, B)
        denom = np.linalg.norm(cross)**2

        if denom < 1e-6:
            # 线段平行
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