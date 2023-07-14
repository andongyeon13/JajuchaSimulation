import Actor
import MyMath as mm

# 포인트 라이트 클래스 (고정)
class PointLight:
    # 생성자 함수
    def __init__(self, position, lightColor, maxDistance = 5):
        self.x, self.y, self.z = tuple(map(lambda a: float(a), position))
        self.r, self.g, self.b = tuple(map(lambda a: float(a), lightColor))
        self.maxDistance = float(maxDistance)

    # x, y, z 좌표 반환
    def getXYZ(self):
        return self.x, self.y, self.z

    # r, g, b 색상 반환
    def getRGB(self):
        return self.r, self.g, self.b

    # x, y, z 좌표 설정
    def setXYZ(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

# 스포트 라이트 클래스 (이동)
class SpotLight(Actor.Actor):
    # 생성자 함수
    def __init__(self, position, lightColor, maxDistance, cutoff, directionVec3, parent=None):
        super().__init__(parent)
        self.pos_l = list(map(lambda a:float(a), position))
        self.lookHorDeg = 90.0
        self.r, self.g, self.b = tuple(map(lambda a:float(a), lightColor))
        self.maxDistance = float(maxDistance)
        self.cutoff_f = float(cutoff)
        self.directionVec3 = directionVec3
        self.directionVec3.normalize()

    # r, g, b 색상 반환
    def getRGB(self):
        return self.r, self.g, self.b

    # x, y, z 좌표 설정
    def setXYZ(self, x, y, z):
        self.pos_l[0] = float(x)
        self.pos_l[1] = float(y)
        self.pos_l[2] = float(z)
