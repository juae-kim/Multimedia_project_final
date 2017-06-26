# 멀티디미디어 최종 프로젝트 <졸음운전 장면 탐지>

## 프로젝트 개요
본 프로그램은 사용자(운전자)의 얼굴 정보를 웹카메라를 통해 실시간으로 입력받아 사용자의 졸음운전 여부를 탐지하는 프로그램 입니다. 사용자의 특정 행위들이 졸음운전으로 탐지되면 alarm을 울려 사용자에게 경각심을 줍니다.

## 실행방법
1. Resorce 디렉토리 내 "warning.wav"파일을 프로젝트에 추가한다.
1. Code 디렉토리 내 소스코드들을 프로젝트에 추가하고 main.cpp를 실행한다.
1. 사용자의 얼굴이 충분히 인식되도록 정방향에 위치한다.
1. 졸음운전 탐지 측정 척도
>- 사용자의 눈 감는 시간이 최대 0.69초 이상일 시 alarm
>- 사용자의 눈 깜빡임이 60초당 48회 이상일 시 alarm
>- 사용자의 입 벌림 정도가 5.6초 이상일 시 하품으로 간주하여 alarm

## 프로그램 실행 요구사항
- 웹카메라 설치
- opencv 2.4.9버전 설치
- 개발언어 C++

## 개발환경
- Visual Studio 2012 64bit 운영체제