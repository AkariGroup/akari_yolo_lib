#!/usr/bin/env python

import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import blobconverter
import cv2
import depthai as dai
import numpy as np

DISPLAY_WINDOW_SIZE_RATE = 2.0
MAX_Z = 15000
idColors = np.random.random(size=(256, 3)) * 256


class TextHelper(object):
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def put_text(self, frame: np.ndarray, text: str, coords: Tuple[int, int]) -> None:
        cv2.putText(
            frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type
        )
        cv2.putText(
            frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type
        )

    def rectangle(
        self, frame: np.ndarray, p1: Tuple[float], p2: Tuple[float], id: int
    ) -> None:
        cv2.rectangle(frame, p1, p2, (0, 0, 0), 4)
        cv2.rectangle(frame, p1, p2, idColors[id], 2)


class HostSync(object):
    def __init__(self, sync_size: int = 4):
        self.dict = {}
        self.head_seq = 0
        self.sync_size = sync_size

    def add_msg(self, name: str, msg: Any, seq: Optional[str] = None) -> None:
        if seq is None:
            seq = str(msg.getSequenceNum())
        if seq not in self.dict:
            self.dict[seq] = {}
        self.dict[seq][name] = msg

    def get_msgs(self) -> Any:
        remove = []
        for name in self.dict:
            remove.append(name)
            if len(self.dict[name]) == self.sync_size:
                ret = self.dict[name]
                for rm in remove:
                    del self.dict[rm]
                return ret
        return None


class OakdTrackingYolo(object):
    def __init__(
        self,
        config_path: str,
        model_path: str,
        fps: int,
        fov: float,
        cam_debug: bool = False,
        robot_coordinate: bool = False,
        track_targets: Optional[List[Union[int, str]]] = None,
    ) -> None:
        if not Path(config_path).exists():
            raise ValueError("Path {} does not poetry exist!".format(config_path))
        with Path(config_path).open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        # parse input shape
        if "input_size" in nnConfig:
            self.width, self.height = tuple(
                map(int, nnConfig.get("input_size").split("x"))
            )

        self.jet_custom = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET
        )
        self.jet_custom = self.jet_custom[::-1]
        self.jet_custom[0] = [0, 0, 0]
        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        self.classes = metadata.get("classes", {})
        self.coordinates = metadata.get("coordinates", {})
        self.anchors = metadata.get("anchors", {})
        self.anchorMasks = metadata.get("anchor_masks", {})
        self.iouThreshold = metadata.get("iou_threshold", {})
        self.confidenceThreshold = metadata.get("confidence_threshold", {})
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
            self.nn_path = str(
                blobconverter.from_zoo(
                    model_path, shaves=6, zoo_type="depthai", use_cache=True
                )
            )
        self.fps = fps
        self.fov = fov
        self.cam_debug = cam_debug
        self.track_targets = track_targets
        self.robot_coordinate = robot_coordinate
        self._stack = contextlib.ExitStack()
        self._pipeline = self._create_pipeline()
        self._device = self._stack.enter_context(dai.Device(self._pipeline))
        self.qRgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qDet = self._device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.qRaw = self._device.getOutputQueue(name="raw", maxSize=4, blocking=False)
        self.qDepth = self._device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )
        self.qTrack = self._device.getOutputQueue(
            "tracklets", maxSize=4, blocking=False
        )
        self.counter = 0
        self.startTime = time.monotonic()
        self.frame_name = 0
        self.dir_name = ""
        self.path = ""
        self.num = 0
        self.text = TextHelper()
        if self.robot_coordinate:
            from akari_client import AkariClient

            self.akari = AkariClient()
            self.joints = self.akari.joints
            self.sync = HostSync(5)
        else:
            self.sync = HostSync(4)
        self.track = None
        self.bird_eye_frame = self.create_bird_frame()
        self.raw_frame = None

    def close(self) -> None:
        self._device.close()

    def convert_to_pos_from_akari(self, pos: Any, pitch: float, yaw: float) -> Any:
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

    def get_labels(self) -> None:
        return self.labels

    def _create_pipeline(self) -> dai.Pipeline:
        # Create pipeline
        pipeline = dai.Pipeline()
        device = dai.Device()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewKeepAspectRatio(False)

        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        objectTracker = pipeline.create(dai.node.ObjectTracker)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        trackerOut = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        trackerOut.setStreamName("tracklets")

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setPreviewSize(1920, 1080)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
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

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        monoRight.setFps(self.fps)
        monoLeft.setFps(self.fps)

        spatialDetectionNetwork.setBlobPath(self.nn_path)
        spatialDetectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(300)
        spatialDetectionNetwork.setDepthUpperThreshold(35000)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(self.classes)
        spatialDetectionNetwork.setCoordinateSize(self.coordinates)
        spatialDetectionNetwork.setAnchors(self.anchors)
        spatialDetectionNetwork.setAnchorMasks(self.anchorMasks)
        spatialDetectionNetwork.setIouThreshold(self.iouThreshold)

        # トラッキングする物体のID、もしくは物体名を配列で渡す。
        # 指定がない場合はconfigファイル内の全物体をトラッキング対象に指定
        if self.track_targets is None:
            objectTracker.setDetectionLabelsToTrack(list(range(self.classes)))
        else:
            target_list = []
            for target in self.track_targets:
                if isinstance(target, int):
                    target_list.append(target)
                elif isinstance(target, str):
                    if target in self.labels:
                        target_list.append(self.labels.index(target))
            objectTracker.setDetectionLabelsToTrack(target_list)
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(
            dai.TrackerIdAssignmentPolicy.UNIQUE_ID
        )

        manip.out.link(spatialDetectionNetwork.input)
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        xoutNn = pipeline.create(dai.node.XLinkOut)
        xoutNn.setStreamName("nn")
        spatialDetectionNetwork.out.link(xoutNn.input)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        xoutRaw = pipeline.create(dai.node.XLinkOut)
        xoutRaw.setStreamName("raw")
        camRgb.video.link(xoutRaw.input)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
        spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        spatialDetectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
        objectTracker.out.link(trackerOut.input)
        return pipeline

    def frame_norm(self, frame: np.ndarray, bbox: Tuple[float]) -> List[int]:
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def get_frame(self) -> Union[np.ndarray, List[Any], Any]:
        frame = None
        detections = []
        ret = False
        try:
            ret = self.qRgb.has()
            if ret:
                rgb_mes = self.qRgb.get()
                self.sync.add_msg("rgb", rgb_mes)
                if self.robot_coordinate:
                    self.sync.add_msg(
                        "head_pos",
                        self.joints.get_joint_positions(),
                        str(rgb_mes.getSequenceNum()),
                    )
        except BaseException:
            raise
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
                self.sync.add_msg("raw", self.qRaw.get())
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qDet.has()
            if ret:
                self.sync.add_msg("detections", self.qDet.get())
                self.counter += 1
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qTrack.has()
            if ret:
                self.track = self.qTrack.get()
        except BaseException:
            raise
        msgs = self.sync.get_msgs()
        tracklets = None
        if msgs is not None:
            detections = msgs["detections"].detections
            frame = msgs["rgb"].getCvFrame()
            depthFrame = msgs["depth"].getFrame()
            self.raw_frame = msgs["raw"].getCvFrame()
            depthFrameColor = cv2.normalize(
                depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, self.jet_custom)
            if self.cam_debug:
                cv2.imshow("rgb", cv2.resize(frame, (self.width, self.height)))
                cv2.imshow(
                    "depth",
                    cv2.resize(depthFrameColor, (self.width, int(self.width * 3 / 4))),
                )
            height = int(frame.shape[1] * 9 / 16)
            width = frame.shape[1]
            brank_height = width - height
            frame = frame[
                int(brank_height / 2) : int(frame.shape[0] - brank_height / 2),
                0:width,
            ]
            for detection in detections:
                # Fix ymin and ymax to cropped frame pos
                detection.ymin = (width / height) * detection.ymin - (
                    brank_height / 2 / height
                )
                detection.ymax = (width / height) * detection.ymax - (
                    brank_height / 2 / height
                )
            if self.track is not None:
                tracklets = self.track.tracklets
                for tracklet in tracklets:
                    # Fix roi to cropped frame pos
                    tracklet.roi.y = (width / height) * tracklet.roi.y - (
                        brank_height / 2 / height
                    )
                    tracklet.roi.height = tracklet.roi.height * width / height

            if self.robot_coordinate:
                self.pos = msgs["head_pos"]
                for detection in detections:
                    converted_pos = self.convert_to_pos_from_akari(
                        detection.spatialCoordinates, self.pos["tilt"], self.pos["pan"]
                    )
                    detection.spatialCoordinates.x = converted_pos[0][0]
                    detection.spatialCoordinates.y = converted_pos[1][0]
                    detection.spatialCoordinates.z = converted_pos[2][0]
                for tracklet in tracklets:
                    converted_pos = self.convert_to_pos_from_akari(
                        tracklet.spatialCoordinates, self.pos["tilt"], self.pos["pan"]
                    )
                    tracklet.spatialCoordinates.x = converted_pos[0][0]
                    tracklet.spatialCoordinates.y = converted_pos[1][0]
                    tracklet.spatialCoordinates.z = converted_pos[2][0]
        return frame, detections, tracklets

    def get_raw_frame(self) -> np.ndarray:
        return self.raw_frame

    def get_labeled_frame(
        self,
        frame: np.ndarray,
        tracklets: List[Any],
        id: Optional[int] = None,
        disp_info: bool = False,
    ) -> np.ndarray:
        for tracklet in tracklets:
            if id is not None and tracklet.id != id:
                continue
            if tracklet.status.name == "TRACKED":
                roi = tracklet.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)
                try:
                    label = self.labels[tracklet.label]
                except Exception:
                    label = tracklet.label
                self.text.rectangle(frame, (x1, y1), (x2, y2), tracklet.id)
                if disp_info:
                    self.text.put_text(frame, str(label), (x1 + 10, y1 + 20))
                    self.text.put_text(
                        frame,
                        f"ID: {[tracklet.id]}",
                        (x1 + 10, y1 + 45),
                    )
                    self.text.put_text(frame, tracklet.status.name, (x1 + 10, y1 + 70))

                    if tracklet.spatialCoordinates.z != 0:
                        self.text.put_text(
                            frame,
                            "X: {:.2f} m".format(tracklet.spatialCoordinates.x / 1000),
                            (x1 + 10, y1 + 95),
                        )
                        self.text.put_text(
                            frame,
                            "Y: {:.2f} m".format(tracklet.spatialCoordinates.y / 1000),
                            (x1 + 10, y1 + 120),
                        )
                        self.text.put_text(
                            frame,
                            "Z: {:.2f} m".format(tracklet.spatialCoordinates.z / 1000),
                            (x1 + 10, y1 + 145),
                        )
        return frame

    def display_frame(
        self, name: str, frame: np.ndarray, tracklets: List[Any], birds: bool = True
    ) -> None:
        if frame is not None:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * DISPLAY_WINDOW_SIZE_RATE),
                    int(frame.shape[0] * DISPLAY_WINDOW_SIZE_RATE),
                ),
            )
            if tracklets is not None:
                self.get_labeled_frame(
                    self, frame=frame, tracklets=tracklets, disp_info=True
                )
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(
                    self.counter / (time.monotonic() - self.startTime)
                ),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.3,
                (255, 255, 255),
            )
            # Show the frame
            cv2.imshow(name, frame)
            if birds:
                self.draw_bird_frame(tracklets)

    def create_bird_frame(self) -> np.ndarray:
        fov = self.fov
        frame = np.zeros((300, 300, 3), np.uint8)
        cv2.rectangle(
            frame, (0, 283), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1
        )

        alpha = (180 - fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array(
            [
                (0, frame.shape[0]),
                (frame.shape[1], frame.shape[0]),
                (frame.shape[1], max_p),
                (center, frame.shape[0]),
                (0, max_p),
                (0, frame.shape[0]),
            ]
        )
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def draw_bird_frame(self, tracklets: List[Any], show_labels: bool = False) -> None:
        birds = self.bird_eye_frame.copy()
        global MAX_Z
        max_x = MAX_Z / 2  # mm
        if tracklets is not None:
            for i in range(0, len(tracklets)):
                if tracklets[i].status.name == "TRACKED":
                    pointY = (
                        birds.shape[0]
                        - int(
                            tracklets[i].spatialCoordinates.z
                            / (MAX_Z - 10000)
                            * birds.shape[0]
                        )
                        - 20
                    )
                    pointX = int(
                        tracklets[i].spatialCoordinates.x / max_x * birds.shape[1]
                        + birds.shape[1] / 2
                    )
                    if show_labels:
                        cv2.putText(
                            birds,
                            self.labels[tracklets[i].label],
                            (pointX - 30, pointY + 5),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.5,
                            (0, 255, 0),
                        )
                    cv2.circle(
                        birds,
                        (pointX, pointY),
                        2,
                        idColors[tracklets[i].id],
                        thickness=5,
                        lineType=8,
                        shift=0,
                    )
        cv2.imshow("birds", birds)
