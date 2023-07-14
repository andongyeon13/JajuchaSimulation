import Actor

# 카메라 클래스
class Camera(Actor.Actor):
    # 생성자 함수
    def __init__(self):
        super().__init__(initPos=[0, 2.5, 0])
