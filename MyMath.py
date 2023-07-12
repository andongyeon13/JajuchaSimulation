import numpy as np
from math import sin, cos, pi, acos, sqrt, tan

# 3차원 벡터 클래스
class Vec3:
    # 생성자 함수
    def __init__(self, *args):
        if len(args) == 3:
            x, y, z = args
        elif len(args) == 1:
            x, y, z = args[0].getXYZ()
        else:
            raise ValueError

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # 좌표 문자열 반환
    def __str__(self):
        return "Vec3, x: {:0.2f}, y: {:0.2f}, z: {:0.2f}".format(self.x, self.y, self.z)

    # 백터 합 반환
    def __add__(self, other):
        xSum = self.x + other.x
        ySum = self.y + other.y
        zSum = self.z + other.z

        return Vec3(xSum, ySum, zSum)

    # 백터 차 반환
    def __sub__(self, other):
        xSub = self.x - other.x
        ySub = self.y - other.y
        zSub = self.z - other.z

        return Vec3(xSub, ySub, zSub)

    # 백터 x, y, z 좌표 반환
    def getXYZ(self):
        return self.x, self.y, self.z

    # 백터 x, y, z 좌표 설정
    def setXYZ(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # 벡터 회전
    def rotate(self, degree, x, y, z):
        rotateMat = rotateMat4(degree, x, y, z)
        a = rotateMat.transpose() * np.matrix(self.getXYZ() + (0,), dtype=np.float32).transpose()
        self.x = a.item(0)
        self.y = a.item(1)
        self.z = a.item(2)

    # 단위 벡터 반환
    def normalize(self):
        length = sqrt(self.x**2 + self.y**2 + self.z**2)
        xNew = self.x / length
        yNew = self.y / length
        zNew = self.z / length

        return Vec3(xNew, yNew, zNew)

    # 벡터 외적 반환
    def cross(self, other):
        return Vec3(self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y - self.y*other.x)

    # 벡터 내적 반환
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    # 벡터 각도차 반환
    def radianDiffer(self, other):
        radian = acos(
            (self.dot(other)) /
            ( sqrt(self.x**2 + self.y**2 + self.z**2)*sqrt(other.x**2 + other.y**2 + other.z**2) )
        )
        return Angle.radian(radian)

# 4차원 벡터 클래스
class Vec4(Vec3):
    # 생성자 함수
    def __init__(self, *args):
        if len(args) == 4:
            x, y, z, w = args
            super().__init__(x, y, z)
            self.w = float(w)
        elif len(args) == 2:
            vec3, w = args
            super().__init__(*vec3.getXYZ())
            self.w = float(w)
        else:
            raise ValueError

    # 좌표 문자열 반환
    def __str__(self):
        return "Vec4, x: {:0.2f}, y: {:0.2f}, z: {:0.2f}, w: {:0.2f}".format(self.x, self.y, self.z, self.w)

    # 벡터 합 반환
    def __add__(self, other):
        xSum = self.x + other.x
        ySum = self.y + other.y
        zSum = self.z + other.z
        wSum = self.w + other.w
        if wSum > 0:
            wSum = 1.0

        return Vec4(xSum, ySum, zSum, wSum)

    # 벡터 차 반환
    def __sub__(self, other):
        xSub = self.x - other.x
        ySub = self.y - other.y
        zSub = self.z - other.z
        wSub = self.w = other.w

        return Vec4(xSub, ySub, zSub, wSub)

    # 벡터 곱 반환
    def __mul__(self, other):
        other = float(other)
        xNew = self.x * other
        yNew = self.y * other
        zNew = self.z * other

        return Vec4(xNew, yNew, zNew, self.w)

    # 벡터 길이 반환
    def length(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    # 벡터 x, y, z, w 좌표 반환
    def getXYZW(self):
        return self.x, self.y, self.z, self.w

    # 벡터 x, y, z, w 좌표 설정
    def setXYZW(self, x, y, z, w):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    # 벡터 회전
    def rotate(self, degree, x, y, z):
        rotateMat = rotateMat4(degree, x, y, z)
        a = rotateMat.transpose() * np.matrix(self.getXYZW(), dtype=np.float32).transpose()
        self.x = a.item(0)
        self.y = a.item(1)
        self.z = a.item(2)
        self.w = a.item(3)

    # 단위 벡터 반환
    def normalize(self):
        divider = self.length()
        xNew = self.x / divider
        yNew = self.y / divider
        zNew = self.z / divider
        wNew = self.w / divider

        return Vec4(xNew, yNew, zNew, wNew)

    # 벡터 외적 반환
    def cross(self, other):
        return Vec4(self.y * other.z - self.z * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x,
                    0.0)

    # 벡터 외적 반환 (정규화된 방향 벡터 반환)
    def cross3(self, other, theOther):
        v0 = other - self  # self -> other
        v1 = theOther - other  # other -> theOther

        return v0.cross(v1).normalize()

    # 벡터 내적 반환
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    # 벡터 각도차 반환
    def radianDiffer(self, other):
        radian = acos(
            (self.dot(other)) /
            (sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2) * sqrt(other.x ** 2 + other.y ** 2 + other.z ** 2))
        )
        return Angle.radian(radian)

    # 행렬 변환 반환
    def transform(self, *args):
        a = np.matrix(self.getXYZW(), np.float32)
        for tranMat in args:
            a *= tranMat
        return Vec4(a.item(0), a.item(1), a.item(2), a.item(3))

# 회전 전치 행렬 반환 (백터 회전 연산에 사용)
def rotateMat4(degree, x, y, z):
    radian = degree / 180 * pi
    xa = radian * x
    ya = radian * y
    za = radian * z

    return np.matrix([[cos(ya)*cos(za), cos(xa)*sin(za) + sin(xa)*sin(ya)*cos(za), sin(xa)*sin(za) - cos(xa)*sin(ya)*cos(za), 0.0],
                      [-1*cos(ya)*sin(za), cos(xa)*cos(za) - sin(xa)*sin(ya)*sin(za), sin(xa)*cos(za) + cos(xa)*sin(ya)*sin(za), 0.0],
                      [sin(ya), -1*sin(xa)*cos(ya), cos(xa)*cos(ya), 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32).transpose()

# 벡터 이동
def getMoveVector(speed, lookingAngle):
    radian = lookingAngle.getRadian()
    return speed * cos(radian), speed * sin(radian)

# 3차원 벡터 이동
def getMoveVector3(lookHorAngle, lookVerAngle):
    yVec = lookVerAngle.getDegree() / 90
    if yVec > 1.0 or yVec < -1.0:
        raise FileNotFoundError(yVec)

    divider = 1.0 - abs(yVec)
    xVec, zVec = getMoveVector(1, lookHorAngle)
    xVec *= divider
    zVec *= -divider
    a = Vec3(xVec, yVec, zVec)
    a.normalize()
    return a

# 각도 클래스
class Angle:
    # 초기값 설정
    def __init__(self, degree):
        self.__degree = float(degree)

    # 라디안으로 반환
    @classmethod
    def radian(cls, radian):
        return Angle(radian*180/pi)

    # 각도 문자열 반환
    def __str__(self):
        return "Angle, degree: {}, radian: {}".format(self.getDegree(), self.getRadian())

    # 각도 반환
    def getDegree(self):
        return self.__degree

    # 각도 설정
    def setDegree(self, degree):
        for _ in range(10000):
            if degree >= 360.0:
                degree -= 360.0
            elif degree < 0.0:
                degree += 360.0
            else:
                break
        else:
            raise ValueError("Too big value.")

        self.__degree = float(degree)

    # 라디안 반환
    def getRadian(self):
        return self.getDegree() / 180 * pi

    # 라디안 설정
    def setRadian(self, radian):
        self.setDegree(radian * 180 / pi)

    # 각도 더하기
    def addDegree(self, degree):
        self.setDegree(self.getDegree() + degree)

    # 라디안 더하기
    def addRadian(self, radian):
        self.setRadian(self.getRadian() + radian)

    # 각 복사
    def copy(self):
        return Angle(self.__degree)

# 3차원 벡터 정규화 (단위 벡터 x, y, z 값 반환)
def normalizeVec3(x, y, z):
    length = sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= length
    y /= length
    z /= length

    return x, y, z

# 단위 행렬
def identityMat4():
    return np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)

# 이동 변환 행렬 (x, y, z 만큼 벡터 이동)
def translateMat4(x, y, z, scala=1.0):
    return np.matrix([[1.0, 0.0, 0.0, x*scala],
                      [0.0, 1.0, 0.0, y*scala],
                      [0.0, 0.0, 1.0, z*scala],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32).transpose()

# 크기 변환 행렬 (x, y, z 만큼 스케일 초기화)
def scaleMat4(x, y, z):
    return np.matrix([[x, 0.0, 0.0, 0.0],
                      [0.0, y, 0.0, 0.0],
                      [0.0, 0.0, z, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32).transpose()

# 원근 투영 행렬 (경계)
def frustumMat4(left, right, bottom, top, n, f):
    if right == left or top == bottom or n == f or n < 0.0 or f < 0.0:
        return identityMat4()
    else:
        return np.matrix([[2*n, 0, (right + left) / (right - left), 0],
                          [0, (2 * n) / (top - bottom), (top + bottom) / (top - bottom), 0],
                          [0, 0, (n + f) / (n - f), (2 * n * f) / (n - f)],
                          [0, 0, -1, 1]], dtype=np.float32)

# 원근 투영 행렬 (시야각)
def perspectiveMat4(fov, aspect, n, f):
    fov = fov / 180 * pi
    tanHalfFov = tan(fov / 2)
    return np.matrix([[1 / (aspect*tanHalfFov), 0, 0, 0],
                      [0,1 / tanHalfFov, 0, 0],
                      [0, 0, -1 * (f + n) / (f - n), -1 * (2*f*n) / (f - n)],
                      [0, 0, -1, 0]], dtype=np.float32).transpose()

# 직교 투영 변환 행렬
def orthoMat4(l, r, b, t, n, f):
    return np.matrix([[2 / (r - l), 0, 0, (l+r) / (l-r)],
                      [0, 2 / (t-b), 0, (b+t) / (b-t)],
                      [0, 0, 2 / (n-f), (n+f) / (n-f)],
                      [0, 0, 0, 1]], dtype=np.float32).transpose()

# 뷰 행렬
def getlookatMat4(eye, center, up):
    f = (center - eye).normalize()
    upN = up.normalize()
    s = f.cross(upN)
    u = s.cross(f)
    return np.matrix([
        [ s.x,  u.x,  -f.x,  0.0 ],
        [ s.y,  u.y,  -f.y,  0.0 ],
        [ s.z,  u.z,  -f.z,  0.0 ],
        [ 0.0,  0.0,   0.0,  1.0 ]
    ], np.float32).transpose()

def main():
    # 예제 테스트
    a = Vec3(1, 0, 0)
    print(a)
    b = Vec3(1, 3, 2)
    a = a.__add__(b)
    print(a)

    c = Angle(43)
    print(c)
    c.addDegree(10)
    print(c)

    a.rotate(50, 2, 2, 2)
    print(a)


if __name__ == '__main__':
    main()
