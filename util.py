#!/usr/bin/env python

import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm
from typing import List
import time
from dataclasses import dataclass
from datetime import datetime


def download_file(path: str, link: str) -> None:
    """
    指定されたリンクからファイルをダウンロードする関数。

    Args:
        path (str): ダウンロード先のファイルパス
        link (str): ダウンロード元のリンク

    Raises:
        Exception: ダウンロード中にエラーが発生した場合
    """
    if os.path.exists(path):
        return
    # ディレクトリが存在しない場合は作成
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    try:
        # ファイルをダウンロード
        print(f"{path} doesn't exist. Download from {link}")
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(path, "wb") as f:
            for data in response.iter_content(chunk_size=block_size):
                if data:
                    f.write(data)
                    progress.update(len(data))
        progress.close()
        print(f"{path} download finished.")
    except Exception:
        print(f"Download error")


class TextHelper(object):
    """
    フレームに文字列を描画するクラス

    """

    def __init__(self) -> None:
        """クラスのコンストラクタ"""
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def put_text(self, frame: np.ndarray, text: str, coords: Tuple[int, int]) -> None:
        """
        フレームに文字列を描画する。

        Args:
            frame (np.ndarray): 描画対象の画像フレーム。
            text (str): 描画する文字列。
            coords (Tuple[int, int]): 描画開始位置の座標。

        """
        cv2.putText(
            frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type
        )
        cv2.putText(
            frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type
        )

    def rectangle(
        self,
        frame: np.ndarray,
        p1: Tuple[float],
        p2: Tuple[float],
        color: Tuple[int, int, int],
    ) -> None:
        """
        フレームに矩形を描画する。

        Args:
            frame (np.ndarray): 描画対象の画像フレーム。
            p1 (Tuple[float]): 矩形の開始座標。
            p2 (Tuple[float]): 矩形の終了座標。
            color: (Tuple[int, int, int]): 矩形描画色

        """
        cv2.rectangle(frame, p1, p2, (0, 0, 0), 4)
        cv2.rectangle(frame, p1, p2, color, 2)


class HostSync(object):
    """各フレームのメッセージを同期するクラス。"""

    def __init__(self, sync_size: int = 4):
        """HostSyncクラスの初期化メソッド。

        Args:
            sync_size (int, optional): 同期するメッセージの数。デフォルトは4。

        """
        self.dict = {}
        self.head_seq = 0
        self.sync_size = sync_size

    def add_msg(self, name: str, msg: Any, seq: Optional[str] = None) -> None:
        """メッセージをDictに追加するメソッド。

        Args:
            name (str): メッセージの名前。
            msg (Any): 追加するメッセージ。
            seq (str, optional): メッセージのシーケンス番号。デフォルトはNone。

        """
        if seq is None:
            seq = str(msg.getSequenceNum())
        if seq not in self.dict:
            self.dict[seq] = {}
        self.dict[seq][name] = msg

    def get_msgs(self) -> Any:
        """同期されたメッセージを取得するメソッド。

        Returns:
            Any: 同期されたメッセージ。

        """
        remove = []
        for name in self.dict:
            remove.append(name)
            if len(self.dict[name]) == self.sync_size:
                ret = self.dict[name]
                for rm in remove:
                    del self.dict[rm]
                return ret
        return None


@dataclass
class PosLog:
    time: int
    x: float
    y: float
    z: float


class OrbitData(object):
    def __init__(self, name: str, id: int, pos_log: PosLog):
        self.name: str = name
        self.id: int = id
        self.pos_log: List[PosLog] = []
        self.tmp_pos_log: List[PosLog] = [pos_log]


class OrbitDataList(object):
    def __init__(self, labels: List[str], log_path: str):
        self.LOGGING_INTEREVAL = 0.5
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.data: List[OrbitData] = []
        self.labels: List[str] = labels
        current_time = datetime.now()
        self.file_name = (
            log_path + f"/data_{current_time.strftime('%Y%m%d_%H%M%S')}.csv"
        )

    def get_cur_time(self) -> float:
        return time.time - self.start_time()

    def get_orbit_from_id(self, id: int) -> OrbitData:
        for data in self.data:
            if data.id == id:
                return data
        return None

    def add_new_data(self, tracklet: Any):
        pos_data = PosLog(time=self.get_cur_time(), x=tracklet.x, y=tracklet.y, z=tracklet.z)
        self.data.append(OrbitData(name=self.labels[tracklet.label], id=tracklet.id, pos_log=pos_data))

    def add_track_data(self, tracklet: Any, pos_list: List[PosLog]):
        pos_data = PosLog(time=self.get_cur_time(), x=tracklet.x, y=tracklet.y, z=tracklet.z)
        pos_list.append(pos_data)

    def add_orbit_data(self, tracklets: List[Any]) -> None:
        if tracklets is None:
            return
        for tracklet in tracklets:
            new_data = True
            for data in self.data:
                if tracklet.id == data.id:
                    self.add_track_data(tracklet, data.tmp_pos_log)
                    new_data = False
            if new_data:
                self.add_new_data(tracklet)

    def weighted_average_position(pos_logs: List[PosLog], target_time: float) -> PosLog:
        """
        指定された時間での位置を、重み付け平均から推測する。

        Args:
            pos_logs (List[PosLog]): 時間と位置のログのリスト。
            target_time (float): 推測したい時間。

        Returns:
            PosLog: 推測された位置。
        """
        total_weight = 0.0
        weighted_sum_x = 0.0
        weighted_sum_y = 0.0
        weighted_sum_z = 0.0
        for log in pos_logs:
            # 時間差に基づいた重みを計算（時間差が小さいほど重みが大きくなる）
            weight = 1 / (abs(log.time - target_time) + 1e-9)  # ゼロ除算を避けるための微小値
            total_weight += weight
            weighted_sum_x += log.x * weight
            weighted_sum_y += log.y * weight
            weighted_sum_z += log.z * weight
        # 重み付け平均を計算
        avg_x = weighted_sum_x / total_weight
        avg_y = weighted_sum_y / total_weight
        avg_z = weighted_sum_z / total_weight

        return PosLog(time=target_time, x=avg_x, y=avg_y, z=avg_z)

    def fix_pos_log(self) -> None:
        cur_time = self.get_cur_time()
        while True:
            next_time = self.last_update_time + self.LOGGING_INTEREVAL * 3/2
            if cur_time - next_time < 0:
                break
            for data in self.data:
                while True:
                    tmp_list: List[PosLog]=[]
                    if data.tmp_pos_log[0].time < next_time:
                        tmp_list.append(data.tmp_pos_log.pop(0))
                    else:
                        break
                if tmp_list is not None:
                    data.pos_log.append(self.weighted_average_position(tmp_list, self.last_update_time + self.LOGGING_INTEREVAL))
            self.last_update_time += self.LOGGING_INTEREVAL

