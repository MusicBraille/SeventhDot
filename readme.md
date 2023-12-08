# SeventhDot
## 저희끼리도 개발 환경이 맞지 않으면 자꾸 오류가 났던 실험인지라, 안되면 꼭 말씀해주시면 감사하겠습니다!. 혹시 돌아가지 않으면 말씀 주시면 실물 시연도 가능합니다!!!!!!!


1. mxl to text
- mxl2text 폴더의 mxl2txt_simple.ipynb 파일을 실행하시면 악보를 텍스트로 변환할 수 있습니다. 악보명, 저장 텍스트 파일명을 바꾸고 실행하면 다양한 악보에 활용 가능합니다.  
사용 모듈 목록:
- python music21

2. Mobile App
- 안드로이드 환경에서 다운로드 가능
 * talking scores 사용
 1) 텍스트 누르면 읽어줌
 2) 악보 한 음씩 이동 가능
 3) 음원 재생 가능
사용 모듈 목록:
- android.media.MediaPlayer

 
 3. 실험: 실험 전사 파일 및 데이터 정리 파일.

 4. preprocessing
    1). preprocessing 시연
    - 돌리면 악보 파일이 나옵니다. resources 폴더에 접근해야 하기에, 꼭 같은 파일에 넣어놓은 채로 실험해주세요! 

    2) 해당 전처리 과정이 모델 성능을 개선하는지에 대한 실험
    - google 코랩에서 colab_test.ipynb 실험
    * 중요! 함꼐 올라온 bbbox파일과 archive 파일을 구글 드라이브에 올린 뒤 해당 경로를 코드에 정확히 설정한 뒤 실험해주세요.
 
     * 구글 코랩 환경을 CPU -> GPU로 변경 후 실험해주세요. 
사용 모듈 목록:
- openCV
- Tenserflow Keras


