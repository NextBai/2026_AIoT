"""
IoTtalk 聲紋開關介接模組 (IoTtalk Voice Trigger Bridge)
=====================================================
功能說明：
  本模組將 ECAPA-TDNN 聲紋特徵辨識輸出轉換為物聯網平台（如 IoTtalk）的開關訊號。
  - 當授權目標說話者（預設: '蕭'）說話且置信度達標時，反轉當前狀態 (Toggle):
    - 啟動狀態 (ON)  -> 輸出數值 500
    - 關閉狀態 (OFF) -> 輸出數值 0
  - 提供多執行緒佇列輪詢 (pre()) 與同步即時調用 (pre2()) 雙模式。
  - 修復 Bug: 修正 result_queue 佇列存取懸空、加入 auto_start 防阻塞與安全置信度閾值檢驗。
"""

import threading
import queue
import time
import sys
from inference import VoiceIdentifier

# -------------------------------------------------------------
# 全域配置與通訊佇列
# -------------------------------------------------------------
DEFAULT_TARGET_SPEAKER = '蕭'
DEFAULT_CONFIDENCE_THRESHOLD = 70.0  # 門檻: 信心度需 >= 70% 避免誤觸

result_queue = queue.Queue()
status = False  # False: 0 (OFF), True: 500 (ON) (相容原全域命名)
_global_identifier = None


def get_shared_identifier():
    """單例模式取得辨識器，避免重複載入模型節省記憶體與時間"""
    global _global_identifier
    if _global_identifier is None:
        _global_identifier = VoiceIdentifier(use_denoise=True)
    return _global_identifier


def handle_result(result):
    """
    回呼處理函式：將推論結果放入 thread-safe 佇列供主程式取用
    """
    if result is not None:
        result_queue.put(result)
    return result


def run_voice_recognition(auto_start=True, duration_sec=4.5):
    """
    背景工作線程：執行錄音並透過 handle_result 回報
    """
    try:
        identifier = get_shared_identifier()
        identifier.interactive_mic_mode(
            duration_sec=duration_sec,
            on_result=handle_result,
            auto_start=auto_start
        )
    except Exception as e:
        print(f"[錯誤] 背景語音辨識發生異常: {e}", file=sys.stderr)
        result_queue.put({
            "success": False,
            "speaker": None,
            "confidence": 0.0,
            "error": str(e)
        })


def toggle_actuator_state(speaker, confidence, target_speaker=DEFAULT_TARGET_SPEAKER, min_conf=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    核心開關狀態切換邏輯
    - 只有當 speaker 符合且 confidence 達到閾值時才進行狀態反轉
    - 回傳: (new_signal_value, state_changed, log_message)
    """
    global status

    if speaker == target_speaker:
        if confidence < min_conf:
            msg = f"⚠️ 辨識為目標講者【{speaker}】，但信心度 {confidence:.2f}% 低於安全閾值 {min_conf:.1f}%，維持原狀態"
            val = 500 if status else 0
            return val, False, msg

        # 狀態翻轉 (Toggle)
        status = not status
        signal = 500 if status else 0
        status_text = "開啟 (ON, 500)" if status else "關閉 (OFF, 0)"
        msg = f"🔓 成功驗證授權講者【{speaker}】(信心度 {confidence:.2f}%) -> 狀態切換為: {status_text}"
        return signal, True, msg
    else:
        signal = 500 if status else 0
        msg = f"🔒 偵測到說話者【{speaker}】(非授權目標【{target_speaker}】)，維持狀態: {signal}"
        return signal, False, msg


# -------------------------------------------------------------
# 模式 1: 多執行緒背景輪詢架構 (相容原 PR 命名 pre())
# -------------------------------------------------------------
def pre(target_speaker=DEFAULT_TARGET_SPEAKER, min_conf=DEFAULT_CONFIDENCE_THRESHOLD, timeout=15.0):
    """
    背景多執行緒監聽模式：
    啟動背景收音線程，主程序輪詢結果佇列，若辨識到目標講者則回傳開關數值 (500 / 0)。
    """
    global status

    # 清空可能殘留的舊訊號
    while not result_queue.empty():
        result_queue.get_nowait()

    worker = threading.Thread(
        target=run_voice_recognition,
        kwargs={"auto_start": True},
        daemon=True
    )
    worker.start()
    print(f"🎙️ [IoTtalk Bridge] 聲紋辨識已在背景啟動 (目標講者: {target_speaker}, 門檻: {min_conf}%)")

    start_time = time.time()
    while True:
        try:
            # 輪詢佇列
            if not result_queue.empty():
                latest_result = result_queue.get_nowait()
                if latest_result and latest_result.get("success", True):
                    speaker = latest_result.get("speaker")
                    confidence = latest_result.get("confidence", 0.0)

                    signal, changed, msg = toggle_actuator_state(
                        speaker, confidence, target_speaker=target_speaker, min_conf=min_conf
                    )
                    print(f"[IoTtalk 訊號] {msg}")
                    return signal

            # 逾時防禦
            if timeout and (time.time() - start_time > timeout):
                print(f"⏱️ [IoTtalk 警告] 錄音等待逾時 ({timeout}s)，維持現狀")
                return 500 if status else 0

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[IoTtalk] 主程式接收到中斷訊號，停止監聽")
            break

    return 500 if status else 0


# -------------------------------------------------------------
# 模式 2: 單次即時識別模式 (相容原 PR 命名 pre2())
# -------------------------------------------------------------
def pre2(target_speaker=DEFAULT_TARGET_SPEAKER, min_conf=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    同步單次識別模式：
    直接進行一次即時麥克風錄音並推論，若符合目標講者則翻轉狀態並回傳 500 / 0。
    """
    global status
    identifier = get_shared_identifier()
    result = identifier.interactive_mic_mode(
        on_result=handle_result,
        auto_start=False  # 保留終端提示給手動執行
    )

    if not result or not result.get("success", True):
        print("⚠️ 辨識未成功或音訊無效")
        return 500 if status else 0

    speaker = result.get("speaker")
    confidence = result.get("confidence", 0.0)

    signal, changed, msg = toggle_actuator_state(
        speaker, confidence, target_speaker=target_speaker, min_conf=min_conf
    )
    print(f"[IoTtalk 訊號] {msg}")
    return signal


if __name__ == '__main__':
    print("======================================================")
    print("      IoTtalk 聲紋控制單元 (ECAPA-TDNN 500/0 Switch)    ")
    print("======================================================")
    val = pre2()
    print(f"\n👉 最終回傳 IoTtalk 訊號值: {val}")