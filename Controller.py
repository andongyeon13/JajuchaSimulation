import pygame as p
import pygame.locals as pl

# 컨트롤러 클래스 (키보드 입력 및 마우스 입력 담당)
class Controller:
    # 생성자 함수
    def __init__(self, mainLoop, target):
        self.oldState = p.key.get_pressed()  # 이전에 눌린 키
        self.newState = p.key.get_pressed()  # 현재 눌린 키

        self.oldMouseState = p.mouse.get_pressed()  # 이전 마우스
        self.newMouseState = p.mouse.get_pressed()  # 현재 마우스

        self.target = target
        self.mainLoop = mainLoop

        self.mouseControl = False  # 마우스 컨트롤 사용 여부
        self.setMouseControl(self.mouseControl)  # 마우스 컨트롤 사용 여부 (생성자 함수 아래 참고)

        self.target.physics = True  # 물리법칙 적용 여부
        self.mainLoop.camera.pos_l = [41.5, 2.1, 0]  # 카메라 좌표
        self.mainLoop.camera.lookHorDeg = 0.0  # 카메라 수평 시선 각도
        self.mainLoop.camera.lookVerDeg = 0.0  # 카메라 수직 시선 각도
        self.target.renderFlag = False  # 렌더링 여부

    # 업데이트 함수
    def update(self, frameTerm):
        self.oldState = self.newState
        self.newState = p.key.get_pressed()

        self.oldMouseState = self.newMouseState
        self.newMouseState = p.mouse.get_pressed()

        if self.getStateChange(pl.K_ESCAPE) == 1:  # 시뮬레이션 종료
            p.quit()
            sys.exit(0)
        if self.getStateChange(pl.K_c) == 1:  # 마우스 사용 여부 전환
            if self.mouseControl:
                self.setMouseControl(False)
            else:
                self.setMouseControl(True)

        # 카메라 이동
        self.target.move(frameTerm, self.newState[pl.K_w], self.newState[pl.K_s], self.newState[pl.K_a], self.newState[pl.K_d], self.newState[pl.K_SPACE], self.newState[pl.K_LSHIFT])
        # 카메라 시야각 조절 (By 키보드)
        self.target.rotate(frameTerm, self.newState[pl.K_UP], self.newState[pl.K_DOWN], self.newState[pl.K_LEFT], self.newState[pl.K_RIGHT])

        # 카메라 시야각 조절 (By 마우스)
        if self.mouseControl:
            xMouse, yMouse = p.mouse.get_rel()
            if abs(xMouse) <= 1:
                xMouse = 0
            if abs(yMouse) <= 1:
                yMouse = 0
            self.target.rotateMouse(-xMouse, -yMouse)

    # 키 눌림 감지
    def getStateChange(self, index):
        if self.oldState[index] != self.newState[index]:
            return self.newState[index]
        else:
            return -1

    # 마우스 컨트롤 설정자
    def setMouseControl(self, boolean):
        if boolean:
            self.mouseControl = True
            p.event.set_grab(True)
            p.mouse.set_visible(False)
        else:
            self.mouseControl = False
            p.event.set_grab(False)
            p.mouse.set_visible(True)