import time

# 프레임 매니저 클래스
class FrameManager:
    # 생성자 함수
    def __init__(self, PrintFPS = False):

        self.lastUpdate = 0.0  # 프레임 마지막 업데이트 시각
        self.frameCount = 0  # 출력 프레임 수
        self.frameTerm = 0.0  # 프레임 업데이트 시간 간격
        self.lastUpdateFPS = time.time()  # FPS 마지막 업데이트 시각
        self.FPS = 0  # 현재 FPS
        self.lastFPS = 0  # 마지막으로 측정된 FPS
        self.printFPS = PrintFPS  # FPS 출력 여부
        self.isFreshFPS = False  # FPS 새로고침 여부

    # 프레임 수 접근자
    def getFrameCount(self):
        return self.frameCount

    # 프레임 업데이트 시간 간격 접근자
    def getFrameTerm(self):
        if self.frameTerm > 100.0:
            return 0.0
        else:
            return self.frameTerm

    # FPS 접근자
    def getFPS(self):
        if self.isFreshFPS:
            self.isFreshFPS = False
            return self.lastFPS, True
        else:
            return self.lastFPS, False

    # FPS 출력 여부 설정자
    def setPrintFPS(self, option):
        self.printFPS = bool(option)

    # 업데이트 함수
    def update(self):
        self.frameCount += 1
        self.frameTerm = time.time() - self.lastUpdate
        self.lastUpdate = time.time()
        if self.frameCount > 10000:
            self.frameCount = 0

        self.FPS += 1
        if time.time() - self.lastUpdateFPS > 1.0:
            self.lastFPS = self.FPS
            self.FPS = 0
            self.lastUpdateFPS = time.time()
            self.isFreshFPS = True
            if self.printFPS:
                #print("FPS: ", self.lastFPS)
                pass
