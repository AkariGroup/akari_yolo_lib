#!/usr/bin/env python

import contextlib
import math
import time
from typing import Any, List, Tuple, Union

import cv2
import depthai as dai
import numpy as np

from .calc_spatial import HostSpatialsCalc
from .util import HostSync, TextHelper


class OakdSpatial(object):
    """OAK-Dを使用してrgbとdepthのフレーム、3次元位置の取得を行うクラス。"""

    def __init__(
        self,
        fps: int,
        fov: float = 73.0,
        spatial_delta: int = 5,
        cam_debug: bool = False,
        robot_coordinate: bool = False,
    ) -> None:
        """クラスの初期化メソッド。

        Args:
            fps (int): カメラのフレームレート。
            fov (float): カメラの視野角 (degree)。defaultはOAK-D LiteのHFOVの73.0[deg]。
            cam_debug (bool, optional): カメラのデバッグ用ウィンドウを表示するかどうか。デフォルトはFalse。
            robot_coordinate (bool, optional): ロボットのヘッド向きを使って物体の位置を変換するかどうか。デフォルトはFalse。

        """

        self.width = 640
        self.height = 640
        self.jet_custom = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET
        )
        self.jet_custom = self.jet_custom[::-1]
        self.jet_custom[0] = [0, 0, 0]
        # extract metadata
        self.fps = fps
        self.fov = fov
        self.cam_debug = cam_debug
        self.robot_coordinate = robot_coordinate
        self.max_z = 15000  # [mm]
        self._stack = contextlib.ExitStack()
        self._pipeline = self._create_pipeline()
        self._device = self._stack.enter_context(dai.Device(self._pipeline))
        self.host_spatials = HostSpatialsCalc(
            self._device, fov=self.fov, delta=spatial_delta
        )
        self.host_spatials.setDeltaRoi(spatial_delta)
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.qRaw = self._device.getOutputQueue(name="raw", maxSize=4, blocking=False)
        self.qDepth = self._device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )
        self.counter = 0
        self.start_time = time.monotonic()
        self.frame_name = 0
        self.dir_name = ""
        self.path = ""
        self.num = 0
        self.text = TextHelper()
        if self.robot_coordinate:
            from akari_client import AkariClient

            self.akari = AkariClient()
            self.joints = self.akari.joints
            self.sync = HostSync(3)
        else:
            self.sync = HostSync(2)

    def close(self) -> None:
        """OAK-Dを閉じる。"""
        self._device.close()

    def convert_to_pos_from_akari(self, pos: Any, pitch: float, yaw: float) -> Any:
        """入力されたAKARIのヘッドの向きに応じて、カメラからの三次元位置をAKARI正面からの位置に変換する。

        Args:
            pos (Any): 物体の3次元位置。
            pitch (float): AKARIのヘッドのチルト角度。
            yaw (float): AKARIのヘッドのパン角度。

        Returns:
            Any: 変換後の3次元位置。

        """
        pitch = -1 * pitch
        yaw = -1 * yaw
        cur_pos = np.array([[pos.x], [pos.y], [pos.z]])
        arr_y = np.array(
            [
                [math.cos(yaw), 0, math.sin(yaw)],
                [0, 1, 0],
                [-math.sin(yaw), 0, math.cos(yaw)],
            ]
        )
        arr_p = np.array(
            [
                [1, 0, 0],
                [
                    0,
                    math.cos(pitch),
                    -math.sin(pitch),
                ],
                [0, math.sin(pitch), math.cos(pitch)],
            ]
        )
        ans = arr_y @ arr_p @ cur_pos
        return ans

    def get_labels(self) -> List[Any]:
        """認識ラベルファイルから読み込んだラベルのリストを取得する。

        Returns:
            List[str]: 認識ラベルのリスト。

        """
        return self.labels

    def _create_pipeline(self) -> dai.Pipeline:
        """OAK-Dのパイプラインを作成する。

        Returns:
            dai.Pipeline: OAK-Dのパイプライン。

        """
        # Create pipeline
        pipeline = dai.Pipeline()
        device = dai.Device()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewKeepAspectRatio(False)

        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setPreviewSize(1280, 720)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(self.fps)
        try:
            calibData = device.readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
        except BaseException:
            raise
        # Use ImageMqnip to resize with letterboxing
        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(self.width * self.height * 3)
        manip.initialConfig.setResizeThumbnail(self.width, self.height)
        camRgb.preview.link(manip.inputImage)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        monoRight.setFps(self.fps)
        monoLeft.setFps(self.fps)

        xoutRaw = pipeline.create(dai.node.XLinkOut)
        xoutRaw.setStreamName("raw")
        camRgb.video.link(xoutRaw.input)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)
        return pipeline

    def frame_norm(self, frame: np.ndarray, bbox: Tuple[float]) -> List[int]:
        """画像フレーム内のbounding boxの座標をフレームサイズで正規化する。

        Args:
            frame (np.ndarray): 画像フレーム。
            bbox (Tuple[float]): bounding boxの座標 (xmin, ymin, xmax, ymax)。

        Returns:
            List[int]: フレームサイズで正規化された整数座標のリスト。

        """
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def get_frame(self) -> Union[np.ndarray, np.ndarray, float]:
        """フレーム画像と検出結果を取得する。

        Returns:
            Union[np.ndarray, np.ndarray,float]: RGBフレーム,depthフレーム,タイムスタンプのタプル。

        """
        frame = None
        depth_frame = None
        ret = False
        try:
            ret = self.qDepth.has()
            if ret:
                self.sync.add_msg("depth", self.qDepth.get())
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qRaw.has()
            if ret:
                raw_mes = self.qRaw.get()
                self.sync.add_msg("raw", raw_mes)
                if self.robot_coordinate:
                    self.sync.add_msg(
                        "head_pos",
                        self.joints.get_joint_positions(),
                        str(raw_mes.getSequenceNum()),
                    )
        except BaseException:
            raise
        msgs = self.sync.get_msgs()
        timestamp = time.time() - self.start_time
        if msgs is not None:
            depth_frame = msgs["depth"].getCvFrame()
            frame = msgs["raw"].getCvFrame()
            depthFrameColor = cv2.normalize(
                depth_frame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, self.jet_custom)
            if self.cam_debug:
                cv2.imshow("rgb", frame)
                cv2.imshow("depth", depth_frame)
        return frame, depth_frame, timestamp

    def calc_spatials(self, depth_frame: np.ndarray, x: int, y: int) -> List[float]:
        """物体の3次元位置を計算する。

        Args:
            depthFrame (np.ndarray): depthフレーム。
            x (int): 物体のx座標。
            y (int): 物体のy座標。

        Returns:
            List[float]: 物体の3次元位置。

        """
        spatials, centroid = self.host_spatials.calc_spatials(
            depth_frame, (x, y)
        )  # centroid == x/y in our case
        return [spatials["x"], spatials["y"], spatials["z"]]
