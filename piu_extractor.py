"""
PIU 채보 영상 → CSV 추출 프로그램
YouTube PUMP IT UP 채보 영상에서 노트 데이터를 자동 추출하여 CSV로 저장
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 패널 이름 (더블 10패널)
COLUMN_NAMES = [
    "p1_dl", "p1_ul", "p1_c", "p1_ur", "p1_dr",
    "p2_dl", "p2_ul", "p2_c", "p2_ur", "p2_dr",
]

# ROI 반경 (컬럼 중심으로부터 가로/세로 픽셀)
ROI_HALF_W = 15
ROI_HALF_H = 8

# 노트 감지 임계값 (HSV)
SAT_THRESHOLD = 50   # 채도 최소값
VAL_THRESHOLD = 80   # 밝기 최소값
PIXEL_RATIO = 0.25   # ROI 내 노트 픽셀 비율 최소값

# 홀드 판별 기준 (연속 감지 프레임 수)
HOLD_MIN_FRAMES = 15


# ── 1. 영상 다운로드 ──────────────────────────────────────────────

def download_video(url: str, output_path: str = "video.mp4") -> str:
    """yt-dlp로 YouTube 영상 다운로드 (720p 이하)."""
    if os.path.exists(output_path):
        print(f"[다운로드] 기존 파일 사용: {output_path}")
        return output_path

    import yt_dlp

    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": False,
    }
    print(f"[다운로드] {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(output_path):
        sys.exit("[오류] 다운로드 실패")
    print(f"[다운로드] 완료: {output_path}")
    return output_path


# ── 2. 캘리브레이션 ──────────────────────────────────────────────

def load_calibration(path: str):
    """저장된 캘리브레이션 JSON 로드."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[캘리브레이션] 기존 설정 로드: {path}")
        return data
    return None


def save_calibration(path: str, data: dict):
    """캘리브레이션 결과 JSON 저장."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[캘리브레이션] 설정 저장: {path}")


def run_calibration(video_path: str, start_sec: float, end_sec: float,
                    calib_path: str = "calibration.json") -> dict:
    """matplotlib 인터랙티브 캘리브레이션.

    1) 채보 구간 중간 프레임을 표시
    2) 사용자가 10개 컬럼 위치를 순서대로 클릭
    3) 감지 라인 Y좌표를 클릭
    """
    # 기존 캘리브레이션이 있으면 재사용
    existing = load_calibration(calib_path)
    if existing is not None:
        return existing

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    mid_sec = (start_sec + end_sec) / 2
    mid_frame = int(mid_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        sys.exit("[오류] 캘리브레이션 프레임 읽기 실패")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    clicks = []

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        clicks.append((x, y))
        n = len(clicks)

        if n <= 10:
            ax.plot(x, y, "ro", markersize=8)
            ax.annotate(COLUMN_NAMES[n - 1], (x, y),
                        textcoords="offset points", xytext=(0, 12),
                        color="red", fontsize=8, ha="center")
            if n == 10:
                ax.set_title("모든 컬럼 선택 완료! 이제 감지 라인 Y좌표를 클릭하세요",
                             fontsize=11)
        elif n == 11:
            ax.axhline(y=y, color="cyan", linewidth=1, linestyle="--")
            ax.set_title(f"감지 라인 Y={y} 설정 완료! 창을 닫으세요", fontsize=11)

        fig.canvas.draw()

        if n >= 11:
            fig.canvas.mpl_disconnect(cid)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.imshow(frame_rgb)
    ax.set_title("10개 컬럼 위치를 순서대로 클릭 (P1_DL → P2_DR)", fontsize=11)
    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(clicks) < 11:
        sys.exit("[오류] 캘리브레이션 미완료 (클릭 부족)")

    col_positions = [c[0] for c in clicks[:10]]
    detect_y = clicks[10][1]

    data = {
        "columns_x": col_positions,
        "detect_y": detect_y,
        "roi_half_w": ROI_HALF_W,
        "roi_half_h": ROI_HALF_H,
    }
    save_calibration(calib_path, data)
    return data


# ── 3. 프레임별 노트 감지 ────────────────────────────────────────

def detect_note_in_roi(frame_hsv: np.ndarray, cx: int, cy: int,
                       hw: int, hh: int) -> bool:
    """ROI 영역에서 HSV 채도+밝기 기준으로 노트 존재 여부 판별."""
    h, w = frame_hsv.shape[:2]
    y1 = max(0, cy - hh)
    y2 = min(h, cy + hh)
    x1 = max(0, cx - hw)
    x2 = min(w, cx + hw)

    roi = frame_hsv[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    s_channel = roi[:, :, 1]
    v_channel = roi[:, :, 2]

    mask = (s_channel >= SAT_THRESHOLD) & (v_channel >= VAL_THRESHOLD)
    ratio = np.count_nonzero(mask) / mask.size
    return ratio >= PIXEL_RATIO


def scan_frames(video_path: str, start_sec: float, end_sec: float,
                calib: dict) -> list:
    """채보 구간의 모든 프레임을 스캔하여 각 컬럼의 노트 감지 결과 반환.

    Returns:
        list of (frame_no, timestamp, [bool * 10])
    """
    col_xs = calib["columns_x"]
    det_y = calib["detect_y"]
    hw = calib.get("roi_half_w", ROI_HALF_W)
    hh = calib.get("roi_half_h", ROI_HALF_H)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results = []
    frame_count = end_frame - start_frame + 1
    print(f"[스캔] 프레임 {start_frame}~{end_frame} ({frame_count}프레임, {fps:.1f}fps)")

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame_no = start_frame + i
        timestamp = frame_no / fps

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = [
            detect_note_in_roi(frame_hsv, cx, det_y, hw, hh)
            for cx in col_xs
        ]
        results.append((frame_no, timestamp, detections))

        if (i + 1) % 500 == 0 or i == frame_count - 1:
            pct = (i + 1) / frame_count * 100
            print(f"  진행: {i + 1}/{frame_count} ({pct:.1f}%)")

    cap.release()
    print(f"[스캔] 완료: {len(results)}프레임 처리")
    return results


# ── 4. 후처리: 노트 이벤트 변환 ──────────────────────────────────

def postprocess(scan_results: list, hold_min: int = HOLD_MIN_FRAMES) -> list:
    """연속 감지 구간을 노트 이벤트로 변환.

    Returns:
        list of dict: {time_sec, p1_dl, ..., p2_dr}
        값: 0=없음, 1=탭, 2=홀드시작, 3=홀드끝
    """
    num_cols = 10
    events = []  # (timestamp, col_index, event_type)

    for col_idx in range(num_cols):
        # 연속 감지 구간 추출
        segments = []
        seg_start = None

        for i, (frame_no, ts, detections) in enumerate(scan_results):
            detected = detections[col_idx]
            if detected and seg_start is None:
                seg_start = i
            elif not detected and seg_start is not None:
                segments.append((seg_start, i - 1))
                seg_start = None

        # 마지막 구간 마무리
        if seg_start is not None:
            segments.append((seg_start, len(scan_results) - 1))

        # 구간 → 이벤트 변환
        for s, e in segments:
            duration = e - s + 1
            ts_start = scan_results[s][1]
            ts_end = scan_results[e][1]

            if duration >= hold_min:
                # 홀드
                events.append((ts_start, col_idx, 2))  # 홀드 시작
                events.append((ts_end, col_idx, 3))    # 홀드 끝
            else:
                # 탭 (시작 시점에 기록)
                events.append((ts_start, col_idx, 1))

    # 시간순 정렬
    events.sort(key=lambda x: x[0])

    # 같은 타이밍의 이벤트를 하나의 행으로 합침
    rows = {}
    for ts, col_idx, etype in events:
        ts_key = round(ts, 4)
        if ts_key not in rows:
            rows[ts_key] = [0] * num_cols
        rows[ts_key][col_idx] = etype

    result = []
    for ts_key in sorted(rows.keys()):
        row = {"time_sec": round(ts_key, 3)}
        for i, name in enumerate(COLUMN_NAMES):
            row[name] = rows[ts_key][i]
        result.append(row)

    print(f"[후처리] {len(result)}개 노트 이벤트 생성")
    return result


# ── 5. CSV 출력 ──────────────────────────────────────────────────

def export_csv(events: list, output_path: str = "chart_output.csv"):
    """노트 이벤트를 CSV로 저장."""
    df = pd.DataFrame(events, columns=["time_sec"] + COLUMN_NAMES)
    df.to_csv(output_path, index=False)
    print(f"[출력] CSV 저장: {output_path} ({len(df)}행)")
    return output_path


# ── 메인 ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PIU 채보 영상에서 노트 데이터를 CSV로 추출"
    )
    parser.add_argument("url", help="YouTube 영상 URL")
    parser.add_argument("--start", type=float, required=True,
                        help="채보 시작 시간 (초)")
    parser.add_argument("--end", type=float, required=True,
                        help="채보 끝 시간 (초)")
    parser.add_argument("--output", default="chart_output.csv",
                        help="출력 CSV 파일명 (기본: chart_output.csv)")
    parser.add_argument("--video", default="video.mp4",
                        help="다운로드 영상 파일명 (기본: video.mp4)")
    parser.add_argument("--calib", default="calibration.json",
                        help="캘리브레이션 JSON 파일명 (기본: calibration.json)")
    parser.add_argument("--hold-frames", type=int, default=HOLD_MIN_FRAMES,
                        help=f"홀드 판별 최소 프레임 수 (기본: {HOLD_MIN_FRAMES})")
    parser.add_argument("--sat-threshold", type=int, default=SAT_THRESHOLD,
                        help=f"채도 임계값 (기본: {SAT_THRESHOLD})")
    parser.add_argument("--val-threshold", type=int, default=VAL_THRESHOLD,
                        help=f"밝기 임계값 (기본: {VAL_THRESHOLD})")

    args = parser.parse_args()

    # 임계값 적용
    global SAT_THRESHOLD, VAL_THRESHOLD
    SAT_THRESHOLD = args.sat_threshold
    VAL_THRESHOLD = args.val_threshold

    # Step 1: 영상 다운로드
    video_path = download_video(args.url, args.video)

    # Step 2: 캘리브레이션
    calib = run_calibration(video_path, args.start, args.end, args.calib)
    print(f"[캘리브레이션] 컬럼 X좌표: {calib['columns_x']}")
    print(f"[캘리브레이션] 감지 라인 Y: {calib['detect_y']}")

    # Step 3: 프레임 스캔
    scan_results = scan_frames(video_path, args.start, args.end, calib)

    # Step 4: 후처리
    events = postprocess(scan_results, hold_min=args.hold_frames)

    # Step 5: CSV 출력
    csv_path = export_csv(events, args.output)

    print(f"\n완료! 결과 파일: {csv_path}")


if __name__ == "__main__":
    main()
