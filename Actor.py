import MyMath as mm

# 행위자 클래스
class Actor:
    # 생성자 함수
    def __init__(self, parent = None, initPos = None, initScale = None):

        self.parent = parent

        self.collideModels_l = []
        self.collideActors_l = []

        self.physics = True

        if initPos is None:
            self.pos_l = [0.0, 0.0, 0.0]
        else:
            self.pos_l = list(initPos)
        if initScale is None:
            self.scale_l = [1, 1, 1]
        else:
            self.scale_l = list(initScale)

        self.lookVerDeg = 0.0
        self.lookHorDeg = 0.0

        self.moveSpeed = 10.0
        self.rotateSpeed = 250.0
        self.rotateMouseSpeed = 0.05

        self.flySpeed = 10

    # 업데이트 함수
    def updateActor(self, term):
        if self.physics:
            self.applyGravity(term)

    # 이동 함수
    def Automove(self, term, speed):
        if speed > 0:
            xVec, zVec = mm.getMoveVector(abs(speed) / 15, mm.Angle(self.lookHorDeg + 90))
            self.pos_l[0] += xVec * term
            self.pos_l[2] -= zVec * term
        elif speed < 0:
            xVec, zVec = mm.getMoveVector(abs(speed) / 15, mm.Angle(self.lookHorDeg + 90))
            self.pos_l[0] -= xVec * term
            self.pos_l[2] += zVec * term
        else:
            pass

        self.validateValues()

    # 회전 함수
    def Autorotate(self, term, degree):
        if degree > 0:
            self.lookHorDeg -= abs(degree) * term
        elif degree < 0:
            self.lookHorDeg += abs(degree) * term
        else:
            pass

        self.validateValues()

    # 이동 함수
    def move(self, term, f, b, l, r, u, d):
        if f and not b:
            xVec, zVec = mm.getMoveVector(self.moveSpeed, mm.Angle(self.lookHorDeg + 90))
            self.pos_l[0] += xVec * term
            self.pos_l[2] -= zVec * term
        elif not f and b:
            xVec, zVec = mm.getMoveVector(self.moveSpeed, mm.Angle(self.lookHorDeg + 90))
            self.pos_l[0] -= xVec * term
            self.pos_l[2] += zVec * term

        if l and not r:
            xVec, zVec = mm.getMoveVector(self.moveSpeed, mm.Angle(self.lookHorDeg))
            self.pos_l[0] -= xVec * term
            self.pos_l[2] += zVec * term
        elif not l and r:
            xVec, zVec = mm.getMoveVector(self.moveSpeed, mm.Angle(self.lookHorDeg))
            self.pos_l[0] += xVec * term
            self.pos_l[2] -= zVec * term

        if u and not d:
            self.pos_l[1] += self.flySpeed * term
        elif not u and d:
            self.pos_l[1] -= self.flySpeed * term

        self.validateValues()

    # 이동 함수 (시야 방향)
    def move3(self, term, f, b, l, r, u, d):
        if f and not b:
            moveVec3 = mm.getMoveVector3(mm.Angle(self.lookHorDeg + 90), mm.Angle(self.lookVerDeg))
            self.pos_l[0] += moveVec3.x * self.moveSpeed * term
            self.pos_l[1] += moveVec3.y * self.moveSpeed * term
            self.pos_l[2] += moveVec3.z * self.moveSpeed * term
        elif not f and b:
            moveVec3 = mm.getMoveVector3(mm.Angle(self.lookHorDeg + 90), mm.Angle(self.lookVerDeg))
            self.pos_l[0] -= moveVec3.x * self.moveSpeed_f * term
            self.pos_l[1] -= moveVec3.y * self.moveSpeed_f * term
            self.pos_l[2] -= moveVec3.z * self.moveSpeed_f * term

        if l and not r:
            xVec, zVec = mm.getMoveVector(self.moveSpeed, mm.Angle(self.lookHorDeg))
            self.pos_l[0] -= xVec * term
            self.pos_l[2] += zVec * term
        elif not l and r:
            xVec, zVec = mm.getMoveVector(self.moveSpeed, mm.Angle(self.lookHorDeg))
            self.pos_l[0] += xVec * term
            self.pos_l[2] -= zVec * term

        if u and not d:
            self.pos_l[1] += self.flySpeed * term
        elif not u and d:
            self.pos_l[1] -= self.flySpeed * term

        self.validateValues()

    # 이동함수 (시야 방향)
    def moveForward(self, distance):
        moveVec3 = mm.getMoveVector3(mm.Angle(self.lookHorDeg + 90), mm.Angle(self.lookVerDeg))
        self.pos_l[0] += moveVec3.x * distance
        self.pos_l[1] += moveVec3.y * distance
        self.pos_l[2] += moveVec3.z * distance

    # 회전 함수
    def rotate(self, term, u, d, l, r):
        if u and not d:
            self.lookVerDeg += self.rotateSpeed * term
        elif not u and d:
            self.lookVerDeg -= self.rotateSpeed * term
        if l and not r:
            self.lookHorDeg += self.rotateSpeed * term
        elif not l and r:
            self.lookHorDeg -= self.rotateSpeed * term

        self.validateValues()

    # 회전 함수 (마우스)
    def rotateMouse(self, mx, my):
        self.lookHorDeg += mx * self.rotateMouseSpeed
        self.lookVerDeg += my * self.rotateMouseSpeed * 2

        self.validateValues()

    # 좌표 반환 함수
    def getXYZ(self):
        return tuple(self.pos_l)

    # 지역 좌표 반환 함수
    def getWorldXYZ(self):
        xWorld, yWorld, zWorld = self.pos_l

        if self.parent is not None:
            xWorld += self.parent.pos_l[0]
            yWorld += self.parent.pos_l[1]
            zWorld += self.parent.pos_l[2]

        return xWorld, yWorld, zWorld

    # 지역 각도 반환 함수
    def getWorldDegree(self):
        hor = self.lookHorDeg
        ver = self.lookVerDeg
        if self.parent is not None:
            hor += self.parent.lookHorDeg
            ver += self.parent.lookVerDeg
        return hor, ver

    # 모델의 벡터 반환
    def getLookVec3(self):
        return mm.getMoveVector3(mm.Angle(self.lookHorDeg + 90), mm.Angle(self.lookVerDeg))

    # 모델의 행렬 반환
    def getModelMatrix(self):
        a = mm.scaleMat4(*self.scale_l) * mm.rotateMat4(-1, self.lookVerDeg, self.lookHorDeg, 0)  * mm.translateMat4(*self.pos_l)
        if self.parent is not None:
            a *= self.parent.getModelMatrix()

        return a

    # 중력 적용 함수
    def applyGravity(self, term):
        distance = term * 10
        if distance > 0.5:
            distance = 0.5

        self.pos_l[1] -= distance

    # 시야각 값 제한 함수
    def validateValues(self):
        if self.lookHorDeg > 360:
            self.lookHorDeg %= 360
        elif self.lookHorDeg < 0:
            for _ in range(10):
                self.lookHorDeg += 360
                if self.lookHorDeg >= 0:
                    break

        if self.lookVerDeg > 90:
            self.lookVerDeg = 90
        elif self.lookVerDeg < -90:
            self.lookVerDeg = -90
