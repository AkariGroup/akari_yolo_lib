import math
import numpy as np
import depthai as dai


class HostSpatialsCalc:
    """
    deothフレームから、指定されたROIの平均深度とそのROIの中心座標を計算する。
    """

    def __init__(self, device, fov=None, delta=5, thresh_low=200, thresh_high=30000):
        """
        コンストラクタ

        Args:
            device (depthai.Device): DepthAIデバイスの情報。
            fov (float, optional): DepthAIカメラの水平視野角。デフォルトはNone。
            delta (int, optional): depthの平均を計算する上下左右周辺フレームの幅。デフォルトは5。
            thresh_low (int, optional): ROIの深度の下限[cm]。デフォルトは200。
            thresh_high (int, optional): ROIの深度の上限[cm]。デフォルトは30000。
        """
        if fov is None:
            calibData = device.readCalibration()
            self.mono_hfov = np.deg2rad(calibData.getFov(dai.CameraBoardSocket.LEFT))
        else:
            self.mono_hfov = np.deg2rad(fov)
        # Values
        self.delta = delta
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

    def set_lower_threshould(self, threshold_low):
        """
        深度の下限閾値を設定するメソッド。

        Args:
            threshold_low (int): 深度の下限閾値[cm]。
        """
        self.thresh_low = threshold_low

    def set_upper_threshold(self, threshold_high):
        """
        深度の上限閾値を設定するメソッド。

        Args:
            threshold_high (int): 深度の上限閾値[cm]。
        """
        self.thresh_high = threshold_high

    def set_delta_roi(self, delta):
        """
        ROIのデルタ値を設定するメソッド。

        Args:
            delta (int): ROIの上下左右周辺フレームの幅。
        """
        self.delta = delta

    def _check_input(self, roi, frame):
        """
        入力がROIかポイントかを確認し、ポイントの場合はROIに変換するメソッド。

        Args:
            roi (tuple): ROI (xmin, ymin, xmax, ymax) またはポイント (x, y)。
            frame (np.ndarray): 深度フレーム。

        Returns:
            tuple: ROI (xmin, ymin, xmax, ymax)。

        Raises:
            ValueError: 入力が2値でも4値でもない場合。
        """
        if len(roi) == 4:
            return roi
        if len(roi) != 2:
            raise ValueError(
                "You have to pass either ROI (4 values) or point (2 values)!"
            )
        # Limit the point so ROI won't be outside the frame
        x = min(max(roi[0], self.delta), frame.shape[1] - self.delta)
        y = min(max(roi[1], self.delta), frame.shape[0] - self.delta)
        return (x - self.delta, y - self.delta, x + self.delta, y + self.delta)

    def _calc_angle(self, frame, offset):
        """
        フレームの中心からのオフセットに基づいて角度を計算するメソッド。

        Args:
            frame (np.ndarray): 深度フレーム。
            offset (int): フレーム中心からのオフセット。

        Returns:
            float: 計算された角度（ラジアン）。
        """
        return math.atan(
            math.tan(self.mono_hfov / 2.0) * offset / (frame.shape[1] / 2.0)
        )

    def calc_spatials(self, depthFrame, roi, averaging_method=np.mean):
        """
        深度フレームの指定されたROI内の空間座標を計算するメソッド。

        Args:
            depthFrame (np.ndarray): 深度フレーム。
            roi (tuple): ROI (xmin, ymin, xmax, ymax) またはポイント (x, y)。
            averaging_method (function, optional): ROI内の深度を平均化するメソッド。デフォルトはnp.mean。

        Returns:
            tuple: (空間座標のタプル(x, y, z), ROIの中心点のタプル (x,y))。
                   空間座標の単位は深度フレームの単位と同じ。
        """
        roi = self._check_input(
            roi, depthFrame
        )  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi
        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.thresh_low <= depthROI) & (depthROI <= self.thresh_high)

        averageDepth = averaging_method(depthROI[inRange])

        centroid = (
            int((xmax + xmin) / 2),
            int((ymax + ymin) / 2),
        )
        midW = int(depthFrame.shape[1] / 2)  # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2)  # middle of the depth img height
        bb_x_pos = centroid["x"] - midW
        bb_y_pos = centroid["y"] - midH
        angle_x = self._calc_angle(depthFrame, bb_x_pos)
        angle_y = self._calc_angle(depthFrame, bb_y_pos)

        spatials = (
            averageDepth * math.tan(angle_x),
            -averageDepth * math.tan(angle_y),
            averageDepth,
        )
        return spatials, centroid
