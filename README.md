# Python-C-API-examples

(1) 다음 명령어로 C++ 확장 모듈 설치.<br>
python setup.py build_ext --inplace<br>

(2) 사용 예시:<br>
import numpy as np<br>
import cpytools as ct  &ensp; # 새로 설치한 확장 모듈<br>

a = np.full((5000, 10000), 1.0)<br>

b = ct.rolling_sum(a, 100) &ensp; # 싱글스레드<br>
print(b)<br>

b2 = ct.rolling_sum2(a, 100)  &ensp; # 멀티스레드<br>
print(b2)<br>
