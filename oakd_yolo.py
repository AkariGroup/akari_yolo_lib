#!/usr/bin/env python

import contextlib
import json
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import blobconverter
import cv2
import depthai as dai
import numpy as np

BLACK = (255, 255, 255)
DISPLAY_WINDOW_SIZE_RATE = 2.0


class OakdYolo(object):
    """
    OAK-Dカメラを使用してYOLO物体認識を行うクラス。

    """

    def __init__(self, config_path: str, model_path: str, fps: int = 10) -> None:
        """クラスの初期化コンストラクタ。

        Args:
            config_path (str): 認識ラベルファイルのパス。
            model_path (str): 認識モデルファイルのパス。
            fps (int, optional): カメラのフレームレート。デフォルトは10。

        """
        if not Path(config_path).exists():
            raise ValueError("Path {} does not exist!".format(config_path))
        with Path(config_path).open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        # parse input shape
        if "input_size" in nnConfig:
            self.width, self.height = tuple(
                map(int, nnConfig.get("input_size").split("x"))
            )

        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        self.classes = metadata.get("classes", {})
        self.coordinates = metadata.get("coordinates", {})
        self.anchors = metadata.get("anchors", {})
        self.anchorMasks = metadata.get("anchor_masks", {})
        self.iouThreshold = metadata.get("iou_threshold", {})
        self.confidenceThreshold = metadata.get("confidence_threshold", {})

        print(metadata)
        self.fps = fps
        # parse labels
        nnMappings = config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        self.nn_path = Path(model_path)
        # get model path
        if not self.nn_path.exists():
            print(
                "No blob found at {}. Looking into DepthAI model zoo.".format(
                    self.nn_path
                )
            )
            self.nn_path = Path(
                blobconverter.from_zoo(
                    model_path, shaves=6, zoo_type="depthai", use_cache=True
                )
            )

        self._stack = contextlib.ExitStack()
        self._pipeline = self._create_pipeline()
        self._device = self._stack.enter_context(
            dai.Device(self._pipeline, usb2Mode=True)
        )
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.qControl = self._device.getInputQueue("control")
        self.qRgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qRaw = self._device.getOutputQueue(name="raw")
        self.qDet = self._device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.counter = 0
        self.start_time = time.monotonic()
        self.frame_name = 0
        self.dir_name = ""
        self.path = ""
        self.num = 0
        self.raw_frame = None

    def close(self) -> None:
        """OAK-Dを閉じる。"""
        self._device.close()

    def set_camera_brightness(self, brightness: int) -> None:
        """カメラの明るさを設定する。

        Args:
            brightness (int): 明るさ。デフォルトは0で-10~10の範囲で変更可能。

        """
        ctrl = dai.CameraControl()
        ctrl.setBrightness(brightness)
        self.qControl.send(ctrl)

    def get_labels(self) -> List[str]:
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
        pipeline = dai.Pipeline()

        # Define sources and outputs
        controlIn = pipeline.create(dai.node.XLinkIn)
        camRgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutNn = pipeline.create(dai.node.XLinkOut)
        controlIn.setStreamName("control")
        xoutRgb.setStreamName("rgb")
        xoutNn.setStreamName("nn")

        # Properties
        controlIn.out.link(camRgb.inputControl)
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setPreviewSize(1920, 1080)
        camRgb.setFps(self.fps)

        xoutRaw = pipeline.create(dai.node.XLinkOut)
        xoutRaw.setStreamName("raw")
        camRgb.video.link(xoutRaw.input)

        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(self.width * self.height * 3)  # 640x640x3
        manip.initialConfig.setResizeThumbnail(self.width, self.height)
        camRgb.preview.link(manip.inputImage)

        # Network specific settings
        detectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        detectionNetwork.setNumClasses(self.classes)
        detectionNetwork.setCoordinateSize(self.coordinates)
        detectionNetwork.setAnchors(self.anchors)
        detectionNetwork.setAnchorMasks(self.anchorMasks)
        detectionNetwork.setIouThreshold(self.iouThreshold)
        detectionNetwork.setBlobPath(self.nn_path)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        # Linking
        manip.out.link(detectionNetwork.input)
        detectionNetwork.passthrough.link(xoutRgb.input)
        detectionNetwork.out.link(xoutNn.input)
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

    def get_frame(self) -> Union[np.ndarray, List[Any]]:
        """フレーム画像と検出結果を取得する。

        Returns:
            Union[np.ndarray, List[Any]]: フレーム画像と検出結果のリストのタプル。
        """
        try:
            inRgb = self.qRgb.get()
            inRaw = self.qRaw.get()
            inDet = self.qDet.get()
        except BaseException:
            raise
        if inRgb is not None:
            frame = inRgb.getCvFrame()
        if inRaw is not None:
            self.raw_frame = inRaw.getCvFrame()
        if inDet is not None:
            detections = inDet.detections
            self.counter += 1
            width = frame.shape[1]
            height = frame.shape[1] * 9 / 16
            brank_height = width - height
            frame = frame[
                int(brank_height / 2) : int(frame.shape[0] - brank_height / 2), 0:width
            ]
            for detection in detections:
                # Fix ymin and ymax to cropped frame pos
                detection.ymin = (width / height) * detection.ymin - (
                    brank_height / 2 / height
                )
                detection.ymax = (width / height) * detection.ymax - (
                    brank_height / 2 / height
                )
        return frame, detections

    def get_raw_frame(self) -> np.ndarray:
        """カメラで撮影した生の画像フレームを取得する。

        Returns:
            np.ndarray: 生画像フレーム。
        """
        return self.raw_frame

    def get_labeled_frame(
        self,
        frame: np.ndarray,
        detections: List[Any],
        id: Optional[int] = None,
        disp_info: bool = False,
    ) -> np.ndarray:
        """認識結果をフレーム画像に描画する。

        Args:
            frame (np.ndarray): 画像フレーム。
            detections (List[Any]): 検出結果のリスト。
            id (Optional[int], optional): 描画するオブジェクトのID。指定すると、そのIDのみを描画した画像フレームを返す。指定しない場合は全てのオブジェクトを描画する。
            disp_info (bool, optional): クラス名とconfidenceをフレーム内に表示するかどうか。デフォルトはFalse。

        Returns:
            np.ndarray: 描画された画像フレーム。

        """
        for detection in detections:
            if id is not None and detections.id != id:
                continue
            bbox = self.frame_norm(
                frame,
                (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
            )
            try:
                label = self.labels[detection.label]
            except BaseException:
                label = detection.label
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            if disp_info:
                cv2.putText(
                    frame,
                    label,
                    (bbox[0] + 10, bbox[1] + 20),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
                cv2.putText(
                    frame,
                    f"{int(detection.confidence * 100)}%",
                    (bbox[0] + 10, bbox[1] + 40),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    255,
                )
        return frame

    def display_frame(
        self, name: str, frame: np.ndarray, detections: List[Any]
    ) -> None:
        """画像フレームと認識結果を描画する。

        Args:
            name (str): ウィンドウ名。
            frame (np.ndarray): 画像フレーム。
            detections (List[Any]): 認識結果のリスト。
        """
        if frame is not None:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * DISPLAY_WINDOW_SIZE_RATE),
                    int(frame.shape[0] * DISPLAY_WINDOW_SIZE_RATE),
                ),
            )
            if detections is not None:
                frame = self.get_labeled_frame(
                    frame=frame, detections=detections, disp_info=True
                )
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(
                    self.counter / (time.monotonic() - self.start_time)
                ),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                (255, 255, 255),
            )
            # Show the frame
            cv2.imshow(name, frame)
