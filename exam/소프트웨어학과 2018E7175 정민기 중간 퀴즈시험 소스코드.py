import cv2

title='mouse event'
img = cv2.imread('C:/img/blank_500.jpg') # 도형을 그릴 배경 그림
save_file = 'C:/img/save_image.jpg' # 파일 저장 위치
cv2.imshow(title, img)

colors = {'black':(0, 0, 0), 'red':(0, 0, 255), 'blue':(255, 0, 0), 'green':(0, 255, 0)}

def onChange(x):
    print(x)
    bg = cv2.getTrackbarPos('BG', title)
    img[:] = [bg]
    cv2.imshow(title, img)
    
cv2.createTrackbar('BG', title, 255, 255, onChange) # 배경색을 흰색 ~ 검정색으로 조절할 수 있다.
    
def onMouse(event, x, y, flags, param):
    print(event, x, y, flags)
    color = colors['black']
    if event == cv2.EVENT_LBUTTONDOWN: #마우스 좌클릭을 누른 상태에서
        if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY: #ctrl키 + shift키 누를시
            color = colors['red'] #빨강색 설정
        elif flags & cv2.EVENT_FLAG_CTRLKEY: #ctrl키 누를시
            color = colors['blue'] #파랑색 설정
        elif flags & cv2.EVENT_FLAG_SHIFTKEY: #shift키 누를시
            color = colors['green'] #초록색 설정
        cv2.circle(img, (x, y), 30, color, -1) # 배경에 원을 그림
    elif event == cv2.EVENT_MBUTTONDOWN: #마우스 가운데를 누른 상태에서
        if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY:
            color = colors['red'] 
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            color = colors['blue']
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            color = colors['green']        
        cv2.rectangle(img, (x, y), (x+100, y+100), color, -1) # 배경에 정사각형을 그림
    elif event == cv2.EVENT_RBUTTONDOWN: #마우스 우클릭을 누른 상태에서
        if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY:
            color = colors['red']
        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            color = colors['blue']
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            color = colors['green']
        cv2.line(img, (x, y), (x+100, y+100), color, 5) # 배경에 직선을 그림
    cv2.imshow(title, img)
        
cv2.setMouseCallback(title, onMouse)
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == 27: #esc키 입력시 프로그램 종료
        break
        print('프로그램이 종료되었습니다.')
    elif key == ord('q'): #알바벳 소문자 q키 입력시 그린 이미지 저장
        cv2.imwrite(save_file, img)
        print('이미지가 저장되었습니다.')
    elif key == ord('w'): #알파벳 소문자 w키 입력시 그린 이미지 불러오기
        cv2.imshow('readimage: '+save_file, img)
        print('이미지를 불러왔습니다.')
        
cv2.destroyAllWindows()
