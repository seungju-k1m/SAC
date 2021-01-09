# ML기반 경로 생성

## Intro

![](./docs/images/Intro.gif)

위 프로젝트는  롯데정보통신과 한양대학교 MMC 연구실에서 협업하여, Multi-Agents RL을 이용하여  다수의 모바일 로봇을 제어하고 이를  Unity 환경에서 개발하고 검증하는 프로젝트입니다.<br/> 

참조논문:[Full Distributed Multi-Robot Collision Avoidance via Deep Reinforcement Learning for safe and Efficient Navigation in Complex Scenarios][link]

[link]:https://arxiv.org/abs/1808.03841
## Prerequisites

    Ubuntu 18.04 or MacOs (higher 10.15)

_conda 환경에서 설치하는 것을 추천합니다.!!_

## Installing
    git init

    git remote add origin <this repo>

    git fetch --all

    git checkout main

    pip install -r requirements.txt

## Running the Tests

json format으로 구성된 configuration을 통해 프로그램을 설정할 수 있습니다.

    ./cfg/*.json

프로그램을 효과적으로 사용할 수 있도록 manual을 확인해보세요.

0. [How to Download Unity Environment][downloadLink]

1. [How to Configure the Program][configureLink]

2. [How to manipulate the Environment][manipulateLink]

3. [How to run Following Algorithm][followingLink]

[downloadLink]:https://github.com/Kyushik/Lotte_Mobile_Robot_Project/blob/Last/docs/00_download.md
[followingLink]:https://github.com/Kyushik/Lotte_Mobile_Robot_Project/blob/Last/docs/03_following.md
[manipulateLink]:https://github.com/Kyushik/Lotte_Mobile_Robot_Project/blob/Last/docs/02_manipulatorEnv.md
[configureLink]:https://github.com/Kyushik/Lotte_Mobile_Robot_Project/blob/Last/docs/01_configuration.md

_실물 및 데모 환경_ 에 대한 설정 파일이 준비되었으면

훈련과 테스트는 다음 코드를 통해서 실행할 수 있습니다.

_훈련시_

    python main.py -p [./cfg/*.json] -t

_테스트시_

    python main.py -p [./cfg/*json] -te
