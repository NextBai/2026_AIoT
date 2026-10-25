from inference import VoiceIdentifier
import threading
import queue
import time

result_queue = queue.Queue()

def handle_result(result):
    #result_queue.put(result)
    return result

def run_voice_recognition():
    identifier = VoiceIdentifier()
    identifier.interactive_mic_mode(
        on_result=handle_result
    )


def pre():
    # 背景辨識執行緒
    worker = threading.Thread(
        target=run_voice_recognition,
        daemon=True
    )
    worker.start()

    print("聲紋辨識已在背景啟動")

    status = False
    # 主程式可以繼續執行其他工作
    while True:
        try:
            # 取得最新結果，不等待新結果到來
            latest_result = None

            while not result_queue.empty():
                latest_result = result_queue.get_nowait()

            if latest_result is not None:
                speaker = latest_result["speaker"]
                confidence = latest_result["confidence"]

                #print(f"最新辨識：{speaker}")
                #print(f"信心值：{confidence:.2f}%")
                if speaker == '蕭':
                    if not status:
                        status = True
                        return 500
                    else:
                        status = False
                        return 0

            # 主程式的其他工作
            time.sleep(0.1)

        except KeyboardInterrupt:
            print("主程式結束")
            break
status = False
def pre2():
    global status
    identifier = VoiceIdentifier()
    result = identifier.interactive_mic_mode(
        on_result=handle_result
    )

    if result["speaker"] == '蕭':
        if not status:
            status = True
            return 500
        else:
            status = False
            return 0

if __name__ == '__main__':
    pre2()