import numpy as np
import math


class Camera:
    """
    相机类，用于管理相机位置、朝向和视图变换。
    提供从世界坐标到相机坐标（视图空间）的变换矩阵。
    """

    def __init__(
        self,
        look_from=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        look_at=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        fov=45.0,  # 垂直视场角（度）
        aspect_ratio=1.0,  # 宽高比
        near=0.1,
        far=100.0,
    ):
        """
        初始化相机。

        参数:
            look_from: 相机位置
            look_at: 相机看向的目标点
            up: 世界空间中的上方向
            fov: 垂直视场角（度）
            aspect_ratio: 视口宽高比
            near: 近平面距离
            far: 远平面距离
        """
        self.look_from = np.array(look_from, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.up = self.up / np.linalg.norm(self.up)  # 归一化上向量
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far

        # 计算相机坐标系
        self.update_camera_basis()

        # 初始缓存视图和投影矩阵
        self._view_matrix = None
        self._perspective_matrix = None
        self._orthographic_matrix = None

        # 更新矩阵
        self.update_matrices()

    def update_camera_basis(self):
        """更新相机的基向量（相机坐标系）"""
        # 计算观察方向向量 (z轴方向，指向相机后方)
        self.forward = self.look_at - self.look_from
        # 处理退化情况（相机位置与目标点重合）
        if np.linalg.norm(self.forward) < 1e-8:
            self.forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            self.forward = self.forward / np.linalg.norm(self.forward)

        # 计算右向量 (x轴方向)
        self.right = np.cross(self.forward, self.up)
        # 处理退化情况（forward与up平行）
        if np.linalg.norm(self.right) < 1e-8:
            # 如果forward与up平行，选择一个垂直于forward的向量作为right
            if abs(np.dot(self.forward, np.array([1, 0, 0]))) < 0.9:
                self.right = np.cross(self.forward, np.array([1, 0, 0]))
            else:
                self.right = np.cross(self.forward, np.array([0, 1, 0]))

        self.right = self.right / np.linalg.norm(self.right)

        # 重新计算真正的上向量 (y轴方向)，确保正交
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)

    def update_matrices(self):
        """更新所有变换矩阵"""
        self._view_matrix = self._compute_view_matrix()
        self._perspective_matrix = self._compute_perspective_matrix()
        self._orthographic_matrix = self._compute_orthographic_matrix()

    def _compute_view_matrix(self):
        """计算视图变换矩阵 (世界 -> 相机空间)"""
        # 确保相机基向量已更新
        self.update_camera_basis()

        # 旋转矩阵：将世界坐标系旋转到相机坐标系
        # 每一行是相机坐标系的一个基向量在世界坐标系中的表示
        rot = np.array(
            [
                [self.right[0], self.right[1], self.right[2], 0],
                [self.up[0], self.up[1], self.up[2], 0],
                [
                    -self.forward[0],
                    -self.forward[1],
                    -self.forward[2],
                    0,
                ],  # 注意：Z轴是forward的负方向
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # 平移矩阵：将相机位置移动到原点
        trans = np.array(
            [
                [1, 0, 0, -self.look_from[0]],
                [0, 1, 0, -self.look_from[1]],
                [0, 0, 1, -self.look_from[2]],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # 视图矩阵 = 旋转 * 平移
        view_matrix = rot @ trans
        return view_matrix

    def _compute_perspective_matrix(self):
        """计算透视投影矩阵"""
        # 计算视锥体参数
        fovy_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fovy_rad / 2)
        near, far = self.near, self.far

        # 构建透视投影矩阵 (OpenGL风格)
        return np.array(
            [
                [f / self.aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    def _compute_orthographic_matrix(self):
        """计算正交投影矩阵"""
        # 正交投影中，视锥体大小不依赖于near或far
        # 直接使用fov来确定尺寸
        fovy_rad = math.radians(self.fov)
        # 确定正交投影的高度 (在z=-1平面上的高度)
        top = math.tan(fovy_rad / 2)  # 这里不乘以near
        right = top * self.aspect_ratio
        near, far = self.near, self.far

        # 标准正交投影矩阵 (OpenGL风格)
        return np.array(
            [
                [1 / right, 0, 0, 0],
                [0, 1 / top, 0, 0],
                [
                    0,
                    0,
                    -2 / (far - near),
                    -(far + near) / (far - near),
                ],  # 修正Z轴变换系数符号
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def get_view_matrix(self):
        """获取视图变换矩阵"""
        if self._view_matrix is None:
            self._view_matrix = self._compute_view_matrix()
        return self._view_matrix

    def get_projection_matrix(self, projection_type="perspective"):
        """获取投影矩阵"""
        if projection_type.lower() == "perspective":
            if self._perspective_matrix is None:
                self._perspective_matrix = self._compute_perspective_matrix()
            return self._perspective_matrix
        else:  # 正交投影
            if self._orthographic_matrix is None:
                self._orthographic_matrix = self._compute_orthographic_matrix()
            return self._orthographic_matrix

    # === 相机动画功能 ===

    def yaw(self, angle_degrees):
        """左右摇头 (围绕世界空间y轴旋转)"""
        angle_rad = math.radians(angle_degrees)

        # 计算从look_at到look_from的方向和距离
        direction = self.look_from - self.look_at
        distance = np.linalg.norm(direction)

        # 绕y轴旋转
        rot_matrix = np.array(
            [
                [math.cos(angle_rad), 0, math.sin(angle_rad)],
                [0, 1, 0],
                [-math.sin(angle_rad), 0, math.cos(angle_rad)],
            ],
            dtype=np.float32,
        )

        # 应用旋转
        new_direction = rot_matrix @ direction

        # 更新相机位置
        self.look_from = self.look_at + new_direction

        # 更新相机视图矩阵
        self.update_matrices()
        return self

    def pitch(self, angle_degrees):
        """抬头/低头 (围绕相机的本地x轴旋转)"""
        angle_rad = math.radians(angle_degrees)

        # 确保相机基向量是最新的
        self.update_camera_basis()

        # 获取当前look_from到look_at的向量
        view_vector = self.look_at - self.look_from
        distance = np.linalg.norm(view_vector)

        # 计算旋转轴（相机的右向量）
        rotation_axis = self.right

        # 构建罗德里格斯旋转公式
        # v_rot = v * cos(θ) + (k × v) * sin(θ) + k * (k·v) * (1 - cos(θ))
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        # 将view_vector归一化
        view_dir = view_vector / distance

        # 应用罗德里格斯旋转
        dot_product = np.dot(rotation_axis, view_dir)
        rotated_view = (
            view_dir * cos_angle
            + np.cross(rotation_axis, view_dir) * sin_angle
            + rotation_axis * dot_product * (1 - cos_angle)
        )

        # 确保旋转后的向量长度保持不变
        rotated_view = rotated_view * distance

        # 更新相机位置
        self.look_from = self.look_at - rotated_view

        # 如果旋转到接近垂直时，可能需要调整up向量
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 重置为世界Y轴

        # 更新矩阵
        self.update_matrices()
        return self

    def roll(self, angle_degrees):
        """歪头 (围绕相机的前向轴旋转)"""
        angle_rad = math.radians(angle_degrees)

        # 确保相机基向量是最新的
        self.update_camera_basis()

        # 旋转up向量围绕forward轴
        # 使用罗德里格斯旋转公式
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        rotation_axis = self.forward

        # 将up向量旋转
        dot_product = np.dot(rotation_axis, self.up)
        new_up = (
            self.up * cos_angle
            + np.cross(rotation_axis, self.up) * sin_angle
            + rotation_axis * dot_product * (1 - cos_angle)
        )

        # 更新相机的up向量
        self.up = new_up / np.linalg.norm(new_up)

        # 更新矩阵
        self.update_matrices()
        return self

    def orbit(self, angle_degrees, axis="y"):
        """相机绕目标点旋转"""
        if axis.lower() == "y":
            # 水平轨道运动（左右环绕物体）
            return self.yaw(angle_degrees)
        elif axis.lower() == "x":
            # 垂直轨道运动（上下环绕物体）
            return self.pitch(angle_degrees)
        elif axis.lower() == "z":
            # 倾斜轨道运动
            return self.roll(angle_degrees)
        return self

    def dolly(self, distance):
        """相机沿视线方向移动（推拉摄影机）"""
        # 获取从look_from指向look_at的方向向量
        direction = self.look_at - self.look_from
        if np.linalg.norm(direction) > 1e-8:
            direction = direction / np.linalg.norm(direction)
        else:
            # 如果相机位置与目标点重合，使用默认前方向
            direction = np.array([0, 0, 1], dtype=np.float32)

        # 沿着方向向量移动相机
        self.look_from = self.look_from + direction * distance

        # 更新矩阵
        self.update_matrices()
        return self

    def pan(self, horizontal, vertical):
        """平移相机（同时移动相机位置和目标点）"""
        # 确保相机基向量是最新的
        self.update_camera_basis()

        # 计算移动向量
        move_vector = self.right * horizontal + self.up * vertical

        # 同时移动相机位置和目标点
        self.look_from = self.look_from + move_vector
        self.look_at = self.look_at + move_vector

        # 更新矩阵
        self.update_matrices()
        return self

    def look_at_point(self, target_point):
        """设置相机观察点，保持相机位置不变"""
        self.look_at = np.array(target_point, dtype=np.float32)

        # 更新矩阵
        self.update_matrices()
        return self

    def set_position(self, position):
        """设置相机位置，保持观察方向不变"""
        old_position = self.look_from.copy()
        self.look_from = np.array(position, dtype=np.float32)

        # 移动look_at以保持相同的观察方向
        movement = self.look_from - old_position
        self.look_at = self.look_at + movement

        # 更新矩阵
        self.update_matrices()
        return self

    def transform_vertices(self, vertices, projection_type="perspective"):
        """将顶点从世界空间转换到裁剪空间（应用视图和投影变换）"""
        # 确保顶点数组形状正确
        n_vertices = vertices.shape[0]

        # 转换为齐次坐标
        homogeneous_vertices = np.ones((n_vertices, 4), dtype=np.float32)
        homogeneous_vertices[:, 0:3] = vertices

        # 应用视图变换 (世界 -> 相机空间)
        view_matrix = self.get_view_matrix()
        view_space = homogeneous_vertices @ view_matrix.T

        # 应用投影变换 (相机空间 -> 裁剪空间)
        projection_matrix = self.get_projection_matrix(projection_type)
        clip_space = view_space @ projection_matrix.T

        # 透视除法 (裁剪空间 -> NDC空间)
        # 仅当w分量不接近0时进行除法
        mask = np.abs(clip_space[:, 3]) > 1e-8
        ndc_space = np.zeros_like(clip_space)
        ndc_space[mask, :] = clip_space[mask, :] / clip_space[mask, 3:4]
        ndc_space[~mask, 0:3] = clip_space[~mask, 0:3]

        # 返回NDC坐标
        return ndc_space[:, 0:3], view_space[:, 0:3]
