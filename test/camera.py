import cv2


# the following is the code generated using gpt
def main():
    # 打开默认摄像头（通常是 0 号设备）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("摄像头已打开，按 'q' 键退出...")

    while True:
        # 读取一帧图像
        ret, frame = cap.read()

        if not ret:
            print("无法获取帧")
            break

        # 显示图像
        cv2.imshow('USB Camera', frame)

        # 按 q 键退出循环
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
