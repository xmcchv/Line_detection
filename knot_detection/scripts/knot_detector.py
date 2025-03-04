#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import open3d as o3d
import threading
from sklearn.cluster import DBSCAN

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

        self.text = ""

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

        cv2.imwrite("./rgb.png",rgb_image)
        # 检测线段
        lines = self.detect_lines(rgb_image, depth_image)
        if lines is None:
            return

        # 计算角度和判断打结
        # angles, result_text = self.calculate_angles_and_check_tangles(lines, angle_threshold=40)
        # # 在图像上绘制合并线段和角度
        # for i, (line, angle) in enumerate(zip(lines, angles)):
        #     x1, y1, x2, y2 = line
            
        #     # 绘制合并线段
        #     cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色线段

        #     # 在合并线段的中点处绘制角度文本
        #     mid_x = int((x1 + x2) / 2)
        #     mid_y = int((y1 + y2) / 2)
        #     cv2.putText(rgb_image, f"{angle:.1f} deg", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        #     # 如果有下一条线段，判断两个线段最近距离，如果小于50则绘制延长线并找到交点
            
        #     if i < len(lines) - 1:
        #         next_line = lines[i + 1]
        #         # 计算当前线段和下一条线段的最近距离
        #         distance = self.min_distance_between_lines(
        #             (x1, y1), (x2, y2),
        #             (next_line[0], next_line[1]), (next_line[2], next_line[3])
        #         )
        #         # 判断最近距离是否小于 50
        #         if distance < 50:
        #             # 延长当前线段
        #             extended_line1 = (x1, y1, x1 + (x2 - x1) * 10, y1 + (y2 - y1) * 10)
        #             extended_line2 = (next_line[0], next_line[1], next_line[2] + (next_line[2] - next_line[0]) * 10, next_line[3] + (next_line[3] - next_line[1]) * 10)
                    
        #             # 绘制延长线
        #             cv2.line(rgb_image, (extended_line1[0], extended_line1[1]), (extended_line1[2], extended_line1[3]), (255, 0, 0), 1)  # 红色延长线
        #             cv2.line(rgb_image, (extended_line2[0], extended_line2[1]), (extended_line2[2], extended_line2[3]), (255, 0, 0), 1)  # 红色延长线

        #             # 计算交点
        #             intersection = self.extend_lines_to_intersection(extended_line1, extended_line2)
        #             if intersection:
        #                 cv2.circle(rgb_image, intersection, 5, (0, 0, 255), -1)  # 在交点处绘制实心圆
        #             result_text = "Cable Knotting"
        #         else:
        #             result_text = "Cable Not Knotted"
        #     else:
        #         result_text = "Cable Not Knotted"
        # # 在图像上显示打结结果
        # cv2.putText(rgb_image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # # 显示图像
        # cv2.imshow("Detect Result Draw Lines with Angles", rgb_image)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()

    def calculate_angles_and_check_tangles(self, final_lines, angle_threshold=40):
        all_angles = []  # 存储所有合并线段的锐角

        for line in final_lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # 计算角度
            vertical_angle = (90 - angle) % 180  # 计算与垂直线的角度
            sharp_angle = abs(vertical_angle)  # 取绝对值确保为非负
            
            # 确保锐角在 0 到 90 之间
            if sharp_angle > 90:
                sharp_angle = 180 - sharp_angle  # 计算锐角
            
            all_angles.append(sharp_angle)

        # 判断是否“打结”，如果所有线段的角度都大于阈值
        if len(all_angles) > 0 and all(angle > angle_threshold for angle in all_angles):
            result_text = "Cable Knotting"
        else:
            result_text = "Cable Not Knotted"

        return all_angles, result_text
    

    def extend_lines_to_intersection(self, line1, line2):
        """延长给定的两条线段，返回交点坐标"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # 计算线段的斜率和截距
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1

        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3

        # 计算交点
        det = A1 * B2 - A2 * B1
        if det == 0:
            return None  # 线段平行或重合

        x_inter = (B2 * C1 - B1 * C2) / det
        y_inter = (A1 * C2 - A2 * C1) / det

        return int(x_inter), int(y_inter)

    def show_image(self, image, name):
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def visualize_process(self, steps):
        """显示处理过程各阶段图像"""
        titles = ["Original", "Binary", "line_image_before", "line_image"]
        
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

    # @param titles = ["Original", "Binary", "line_image_before", "line_image"]
    # @param imshow_title "Processing Pipeline"
    def visualize_process_with_titles(self, steps, titles, imshow_title):
        """显示处理过程各阶段图像"""
        # titles = ["Original", "Binary", "line_image_before", "line_image"]
        
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
        
        cv2.imshow(imshow_title, stacked)
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

    def filter_contours_by_bottom_edge(self, binary_image):
        # 1. 找到轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 2. 获取图像的高度
        height = binary_image.shape[0]

        # 3. 存储符合条件的轮廓
        filtered_contours = []

        # 4. 过滤不挨着底边的轮廓
        for contour in contours:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查轮廓底边的 y 坐标是否接近图像底边
            if (height - (y + h)) < 10:  # 10 像素的阈值，可以根据需要调整
                filtered_contours.append(contour)

        return filtered_contours

    def detect_lines(self, image, depth_image):
        height, width, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理（根据实际场景调整阈值）
        # 高斯模糊去噪（调整核大小）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
        
        # 提取轮廓
        contours = self.filter_contours_by_bottom_edge(binary)
        # contour_image = image.copy()  # 创建原始图像的拷贝以绘制轮廓
        # 创建一个空白的二值图像
        contour_image = np.zeros_like(binary)
        # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # 绘制所有轮廓，绿色
        cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)


        lines = cv2.HoughLinesP(
            contour_image,  # 输入细化后的骨架图像
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=250,  # 根据图像尺寸调整
            maxLineGap=20
        )
        # 绘制检测结果
        line_image_before = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image_before, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 合并线段
        print(f'lines: {len(lines) if lines is not None else 0}', end='')
        if lines is not None:
            lines = self.merge_lines(
                lines, 
                angle_thresh=15,  # 降低角度阈值，避免错误合并
                dist_thresh=10, 
                img_width=width, 
                img_height=height, 
                edge_buffer=20,
                img=image.copy()
            )
        print(f' after merge: {len(lines) if lines is not None else 0}')
        
        # 绘制检测结果
        line_image = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # 提取线段坐标
        extracted_lines = []
        if lines is not None:
            for line in lines:
                if len(line) == 4:
                    extracted_lines.append(line)

        
        # 可视化步骤
        steps = [binary, contour_image, line_image_before, line_image]
        self.visualize_process(steps)
        
        return np.array(extracted_lines).reshape(-1, 4) if extracted_lines else np.empty((0, 4))

    
    def merge_lines(self, lines, angle_thresh=10, dist_thresh=20, img_width=640, img_height=480, edge_buffer=20, img=None):
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

        # # 过滤非接近垂直的线段
        # lines = [
        #     line for line in lines
        #     if self.is_near_vertical(line[0], angle_thresh)
        # ]
        # print(f' 过滤非垂直:{len(lines)}', end='')

        # 过滤接近水平的线段
        lines = [
            line for line in lines
            if not self.is_near_horizontal(line[0], angle_thresh)
        ]
        print(f' 过滤水平:{len(lines)}', end='')
        
        if len(lines) == 0:
            return np.array(lines)
        
        img_shape=(img_height, img_width, 3)
        # 计算每条线的角度和中点
        angles = []
        midpoints = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # 计算角度
            midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)  # 计算中点
            angles.append(angle)
            midpoints.append(midpoint)

        angles = np.array(angles)
        midpoints = np.array(midpoints)

        # 将角度归一化到 [0, 360) 范围内
        angles = np.mod(angles, 360)
        angles = angles.reshape(-1, 1)  # 调整形状以适合 DBSCAN

        # 使用 DBSCAN 聚类
        dbscan = DBSCAN(eps=dist_thresh, min_samples=1)  # eps为距离阈值
        labels = dbscan.fit_predict(angles)

        # 为每个聚类分配随机颜色
        unique_labels = np.unique(labels)
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in unique_labels]  # 随机颜色

        # 创建一个空白图像
        # img = np.zeros(img_shape, dtype=np.uint8)

        #判断是否只有两个类别
        if len(unique_labels) == 2:
            #计算两个聚类线段间的最近距离
            cluster1_indices = np.where(labels == 0)[0]
            cluster2_indices = np.where(labels == 1)[0]
            # 计算两个聚类线段间的最近距离
            min_distance = float('inf')
            for i in cluster1_indices:
                for j in cluster2_indices:
                    distance = np.linalg.norm(np.array(midpoints[i]) - np.array(midpoints[j]))
                    if distance < min_distance:
                        min_distance = distance
            print(f" min_distance:{min_distance} ", end='')
            # 如果距离小于阈值，text标位置为"Knotting"
            if min_distance < 420:
                self.text = "Knotting"
            else:
                self.text = "Not Knotting"
            cv2.putText(img, self.text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 如果只有一个聚类，text标位置为"Not Knotting"
            cv2.putText(img, self.text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 绘制线段和合并线段
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 0:
                # 获取属于同一聚类的所有线段
                clustered_lines = [lines[i][0] for i in cluster_indices]
                # 计算聚类线段中y最小的端点和y最大的端点的x,y坐标
                min_y_point = min(clustered_lines, key=lambda line: line[1])[:2]
                max_y_point = max(clustered_lines, key=lambda line: line[3])[2:]
                # 绘制两个端点
                cv2.circle(img, tuple(min_y_point), 5, colors[label], -1)
                cv2.circle(img, tuple(max_y_point), 5, colors[label], -1)
                
                x1 = int(min_y_point[0])
                y1 = int(min_y_point[1])
                x2 = int(max_y_point[0])
                y2 = int(max_y_point[1])

                # 计算x1y1 x2y2的中点坐标
                avg_x = (x1 + x2) / 2
                avg_y = (y1 + y2) / 2
                # 计算x1y1 x2y2的角度
                avg_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # 计算角度
                

                # 添加合并后的线段
                final_lines.append((x1, y1, x2, y2))

                # 绘制聚类的线段
                # for line in clustered_lines:
                #     cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), colors[label], 2)

                # 绘制合并后的线段
                cv2.line(img, (x1, y1), (x2, y2), colors[label], 2)
                # 计算与垂直线的锐角
                vertical_angle = (90 - avg_angle) % 180  # 计算与垂直线的角度
                sharp_angle = abs(vertical_angle)  # 取绝对值确保为非负

                # 确保锐角在 0 到 90 之间
                if sharp_angle > 90:
                    sharp_angle = 180 - sharp_angle  # 计算锐角

                # 在合并线段的中点处绘制角度文本
                text_position = (int(avg_x), int(avg_y))
                cv2.putText(img, f"{sharp_angle:.1f}°", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        # 显示图像
        cv2.imshow("DBSCAN Lines", img)
        cv2.waitKey(1)

        return np.array(final_lines)

    def is_near_vertical(self, line, angle_thresh):
        # 计算线段的角度
        theta = np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi
        # 判断是否接近垂直
        return (90 - angle_thresh) <= abs(theta) <= (90 + angle_thresh)
    
    def is_near_horizontal(self, line, angle_thresh):
        # 计算线段的角度
        theta = np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi
        # 判断是否接近水平
        return (-angle_thresh <= theta <= angle_thresh) or (180 - angle_thresh <= theta <= 180 + angle_thresh)

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
