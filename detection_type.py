import numpy as np
from enum import Enum


class Detection(object):
    """ダミーの検出結果"""

    class SpatialCoordinates(object):
        """ダミーの空間座標"""

        def __init__(self, pos: np.array):
            """コンストラクタ

            Args:
                pos (np.array): 位置
            """
            self.DEFAULT_OBJECT_HEIGHT = 1000  # object_listに追加される物体の高さ
            self.x = pos[0]
            self.y = pos[1]
            self.z = pos[2]

    def __init__(
        self,
        label: str,
        pos: np.array,
        confidence: float = 1.0,
        xmax: float = 0.0,
        xmin: float = 0.0,
        ymax: float = 0.0,
        ymin: float = 0.0,
    ):
        """コンストラクタ

        Args:
            label (str): 物体ラベル
            pos (np.array): 位置
            xmax (float, optional): x座標の最大値。デフォルトは0.0。
            xmin (float, optional): x座標の最小値。デフォルトは0.0。
            ymax (float, optional): y座標の最大値。デフォルトは0.0。
            ymin (float, optional): y座標の最小値。デフォルトは0.0。
        """
        self.confidence = confidence
        self.label = label
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.spatialCoordinates = self.SpatialCoordinates(pos)


class Status(Enum):
    """ステータス"""

    NEW = 1
    TRACKED = 2
    LOST = 3
    REMOVED = 4


class Roi(object):
    """ダミーのROI"""

    class Point(object):
        """ダミーのポイント"""

        def __init__(self, x: float = 0.0, y: float = 0.0):
            """コンストラクタ

            Args:
                x (float, optional): x座標。デフォルトは0.0。
                y (float, optional): y座標。デフォルトは0.0。
            """
            self.x = x
            self.y = y

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
    ):
        """コンストラクタ

        Args:
            x (float, optional): ROIの左上のx座標。デフォルトは0.0。
            y (float, optional): ROIの左上のy座標。デフォルトは0.0。
            width (float, optional): ROIの幅。デフォルトは0.0。
            height (float, optional): ROIの高さ。デフォルトは0.0。
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def denormalize(self, width: int, height: int):
        """ROIをデノーマライズするメソッド

        Args:
            width (int): 画像の幅
            height (int): 画像の高さ
        """
        return Roi(
            self.x * width, self.y * height, self.width * width, self.height * height
        )

    def topLeft(self):
        """depthAIのROIの左上座標を取得するメンバのダミー

        Returns:
            Roi.Point: ROIの左上座標
        """
        return self.Point(self.x, self.y)

    def bottomRight(self):
        """depthAIのROIの右下座標を取得するメンバのダミー

        Returns:
            Roi.Point: ROIの右下座標
        """
        return self.Point(self.x + self.width, self.y + self.height)


class SpatialCoordinates(object):
    """ダミーの空間座標"""

    def __init__(self, pos: np.array):
        """コンストラクタ

        Args:
            pos (np.array): 位置
        """
        self.DEFAULT_OBJECT_HEIGHT = 1000  # object_listに追加される物体の高さ
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]


class Tracklet(object):
    """ダミーのトラックレット"""

    def __init__(
        self,
        id: int,
        label: str,
        pos: np.array,
        roi: Roi,
        last_track_timestamp: float,
        status: Status = Status.NEW,
    ):
        """コンストラクタ

        Args:
            id (int): ID
            label (str): 物体ラベル
            pos (np.array): 位置
            roi (Roi): ROI
            status (str): ステータス
            time (float): 現在のタイムスタンプ
        """
        self.status = status
        self.id = id
        self.label = label
        self.roi = roi
        self.age = 0
        self.spatialCoordinates = SpatialCoordinates(pos)
        self.lastTrackTimestamp = last_track_timestamp

    def update_spatial_coordinates(self, pos: np.array):
        """空間座標を更新するメソッド

        Args:
            pos (np.array): 位置
        """
        self.spatialCoordinates = SpatialCoordinates(pos)
