import cv2
from datetime import datetime
from threading import Thread


class CountsPerSec:

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def counterPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time


class VideoGet:

    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def get(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.stream.read()

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def stop(self):
        self.stopped = True


def putIterationsPerSec(frame, iterations_per_sec):
    cv2.putText(frame, f"{round(iterations_per_sec)} iterations/sec",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    return frame


def noThreading(source=0):
    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()

    while True:
        _, frame = cap.read()

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        frame = putIterationsPerSec(frame, cps.counterPerSec())
        cv2.imshow('Video', frame)
        cps.increment()


def threadVideo(src1=0, src2=1):
    video_getter = VideoGet(src1).start()
    video_getter2 = VideoGet(src2).start()
    cps = CountsPerSec().start()
    cps2 = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame2 = video_getter2.frame
        frame = putIterationsPerSec(frame, cps.counterPerSec())
        frame2 = putIterationsPerSec(frame2, cps2.counterPerSec())
        cv2.imshow("Video1", frame)
        cv2.imshow("Video2", frame2)
        cps.increment()
        cps2.increment()


# noThreading(0)
# noThreading(1)

threadVideo(0, 1)
