# ML기반 경로 생성

## Intro

![](./docs/Intro.gif)

위 프로젝트는  롯데정보통신과 한양대학교 MMC 연구실에서 협업하여, Multi-Agents RL을 이용하여  다수의 모바일 로봇을 제어하고 이를  Unity 환경에서 개발하고 검증하는 프로젝트입니다.

## Prerequisites

    Ubuntu 18.04 or MacOs (higher 10.15)

_conda 환경에서 설치하는 것을 추천합니다.!!_

## Installing

    pip install -r requirements.txt

## Running the Tests

json format으로 구성된 configuration을 통해 프로그램을 설정할 수 있습니다.

    ./cfg/*.json

자세한 설명은 여기를 참조하세요.

[How to Configure the Program][configureLink]

[configureLink]:[./docs/configuration.md]

프로그램 설정 파일이 준비되었으면 훈련과 테스트는 다음 코드를 통해서 실행할 수 있습니다.

_훈련시_

    python main.py -p [파일의 위치] -t

_테스트시_

    python main.py -p [파일의 위치] -te
