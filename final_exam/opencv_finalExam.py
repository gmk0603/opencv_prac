#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
import tkinter.messagebox as msgbox
import cv2
import numpy as np
import matplotlib.pylab as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

window = Tk()
window.title("이미지 편집 프로그램")
window.geometry("675x700")

window.resizable(width=False, height=False)

mainmenu = tk.Menu(window)

def test():
    print("Menu Pressed")
    
def close_window():
    window.destroy()
    
def file_choose():
    window.file = filedialog.askopenfile(
    initialdir='path',
        title='select file',
        filetypes=(('all files', '*.*'), ('png files', '*.png'), ('jpg files','*.jpg')))
    print(window.file.name)
    
    get_path.delete(0, END) 
    #해당 코드가 없으면 이미지를 불러올 때마다 경로 입력창 뒤에 계속해서 새로 선택한 이미지 경로가 붙는다.
    
    get_path.insert(0, window.file.name)
    float_image()

def float_image():
    path = get_path.get()
    photo = Image.open(path)
    resize_photo = photo.resize((300, 300))#미리보기 이미지 사이즈 조정
    final_photo = ImageTk.PhotoImage(resize_photo, master = window)
    image_label.configure(image = final_photo)
    image_label.image = final_photo
    
def insert_figure(): #합성된 이미지를 불러와서 도형 그리기
    read_img = cv2.imread('C:/Temp/save_image.jpg')
    save_file = 'C:/Temp/save_image.jpg' # 파일 저장 및 열기 위치
    cv2.imshow(save_file, read_img)
    
    msgbox.showinfo("편집방법", "자동 저장\n좌클릭: 원, 우클릭: 선, 가운데클릭: 정사각형\n+\nctrl+shift키: 빨강색, ctrl키: 파랑색, shift키: 초록색")    
        
    colors = {'black':(0, 0, 0), 'red':(0, 0, 255), 'blue':(255, 0, 0), 'green':(0, 255, 0)}
   
    def onMouse(event, x, y, flags, param):
        print(event, x, y, flags)
        color = colors['black']
        if event == cv2.EVENT_LBUTTONDOWN: 
            if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY: 
                color = colors['red'] 
            elif flags & cv2.EVENT_FLAG_CTRLKEY: 
                color = colors['blue'] 
            elif flags & cv2.EVENT_FLAG_SHIFTKEY: 
                color = colors['green'] 
            cv2.circle(read_img, (x, y), 30, color, -1) 
        elif event == cv2.EVENT_MBUTTONDOWN: 
            if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY:
                color = colors['red'] 
            elif flags & cv2.EVENT_FLAG_CTRLKEY:
                color = colors['blue']
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                color = colors['green']        
            cv2.rectangle(read_img, (x, y), (x+100, y+100), color, -1) 
        elif event == cv2.EVENT_RBUTTONDOWN: 
            if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY:
                color = colors['red']
            elif flags & cv2.EVENT_FLAG_CTRLKEY:
                color = colors['blue']
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                color = colors['green']
            cv2.line(read_img, (x, y), (x+100, y+100), color, 5) 
        cv2.imshow(save_file, read_img)
        cv2.imwrite(save_file, read_img)
            
    cv2.setMouseCallback(save_file, onMouse)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27: #esc키 입력시 프로그램 종료
            break
            print('프로그램이 종료되었습니다.') 
        elif key == ord('q'): #알바벳 소문자 q키 입력시 그린 이미지 저장
            cv2.imwrite(save_file, read_img)
            print('이미지가 저장되었습니다.')
    cv2.destroyWindow(save_file)
    
def edit_image():
    image = Image.open("C:/Temp/save_image.jpg")
    resize_image = image.resize((300, 300))#미리보기 이미지 사이즈 조정
    edited_image = ImageTk.PhotoImage(resize_image, master = window)
    editimg_label.configure(image = edited_image)
    editimg_label.image = edited_image  

def chroma_select():
    window.file = filedialog.askopenfile(
    initialdir='path',
        title='select file',
        filetypes=(('all files', '*.*'), ('png files', '*.png'), ('jpg files','*.jpg')))
    print(window.file.name)
    
    editimg1Ent.delete(0, END) 
    #해당 코드가 없으면 이미지를 불러올 때마다 경로 입력창 뒤에 계속해서 새로 선택한 이미지 경로가 붙는다.
    
    editimg1Ent.insert(0, window.file.name)

#크로마키 합성 함수
def chromakey():
    img1 = cv2.imread(editimg1Ent.get())
    img2 = cv2.imread(get_path.get())
    print("함수 실행 후 경로 확인")
    print(editimg1Ent.get())
    print(get_path.get())
    
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    x = (width2 - width1)//2
    y = height2 - height1
    w = x + width1
    h = y + height1
    
    chromakey = img1[:10, :10, :]
    offset = 20
    
    hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
    hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    
    chroma_h = hsv_chroma[:, :, 0]
    lower = np.array([chroma_h.min() - offset, 100, 100])
    upper = np.array([chroma_h.max() + offset, 255, 255])
    
    mask = cv2.inRange(hsv_img, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    roi = img2[y:h, x:w]
    fg = cv2.bitwise_and(img1, img1, mask=mask_inv)
    bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2[y:h, x:w] = fg + bg
    
    cv2.imshow('chromaedit', img2)
    cv2.imwrite('C:/Temp/save_image.jpg', img2)
    cv2.waitKey()
    cv2.destroyAllWindows()

#스티커 합성 이미지 선택 함수
def sticker_select():
    window.file = filedialog.askopenfile(
    initialdir='path',
        title='select file',
        filetypes=(('all files', '*.*'), ('png files', '*.png'), ('jpg files','*.jpg')))
    print(window.file.name)
    
    editimg2Ent.delete(0, END) 
    #해당 코드가 없으면 이미지를 불러올 때마다 경로 입력창 뒤에 계속해서 새로 선택한 이미지 경로가 붙는다.
    
    editimg2Ent.insert(0, window.file.name)

#스티커 합성 함수
def SeamlessClone():
    seamimg1 = cv2.imread(editimg2Ent.get())
    seamimg2 = cv2.imread(get_path.get())
    
    def onMouse(event, x, y, flags, param):
        global x0, y0
        if event == cv2.EVENT_LBUTTONDOWN:
            x0 = x
            y0 = y
            mask = np.full_like(seamimg1, 255)
    
            height, width = seamimg2.shape[:2]
            locate = (x0, y0)
    
            mixed = cv2.seamlessClone(seamimg1, seamimg2, mask, locate, cv2.MIXED_CLONE)            
            cv2.imshow('mixed', mixed)
            cv2.moveWindow('mixed', 500, 100)
            cv2.imwrite('C:/Temp/save_image.jpg', mixed)
            
    cv2.imshow('before', seamimg2)
    cv2.moveWindow('before', 900, 100)
    msgbox.showinfo("편집방법", "삽입할 위치를 좌클릭하십시오")
    cv2.setMouseCallback('before', onMouse)    
    cv2.waitKey()
    cv2.destroyAllWindows()

#모자이크 합성 함수
def mosaic():
    rate = 15
    win_title = 'mosaic'
    img = cv2.imread(get_path.get())
    msgbox.showinfo("편집방법", "원하는 범위를 드래그 한 후 Enter클릭!")
    while True:
        x,y,w,h = cv2.selectROI(win_title, img, False)
        if w and h:
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (w//rate, h//rate))
            
            roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)
            img[y:y+h, x:x+w] = roi
            cv2.imshow(win_title, img)
            cv2.imwrite('C:/Temp/save_image.jpg', img)
        else:
            break
    cv2.destroyAllWindows()

#리퀴파이 합성 함수
def liquifys():
    msgbox.showinfo("편집방법", "합성 범위를 계속 드래그!")
    win_title = 'Liquify'
    half = 50
    global isDragging
    isDragging = False

    #리퀴파이 함수
    def liquify(img, cx1, cy1, cx2, cy2):
        x, y, w, h = cx1-half, cy1-half, half*2, half*2

        roi = img[y:y+h, x:x+w].copy()
        out = roi.copy()

        offset_cx1,offset_cy1 = cx1-x, cy1-y
        offset_cx2,offset_cy2 = cx2-x, cy2-y

        tri1 = [[(0, 0), (w, 0), (offset_cx1, offset_cy1)], #상 top
               [[0, 0], [0, h], [offset_cx1, offset_cy1]], #좌, Left
               [[w, 0], [offset_cx1, offset_cy1], [w, h]], #우, right
               [[0, h], [offset_cx1, offset_cy1], [w, h]]] #하, bottom

        tri2 = [[ [0, 0], [w, 0], [offset_cx2, offset_cy2]],
               [[0, 0], [0, h], [offset_cx2, offset_cy2]],
               [[w, 0], [offset_cx2, offset_cy2], [w, h]],
               [[0, h], [offset_cx2, offset_cy2], [w, h]]]

        for i in range(4):
            matrix = cv2.getAffineTransform( np.float32(tri1[i]),
                                           np.float32(tri2[i]))
            warped = cv2.warpAffine( roi.copy(), matrix, (w, h),
                                   None, flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT_101)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(tri2[i]), (255, 255, 255))

            warped = cv2.bitwise_and(warped, warped, mask = mask)
            out = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(mask))
            out = out + warped

        img[y:y+h, x:x+w] = out
        return img

    def onMouse(event, x, y, flags, param):
        global cx1, cy1, img, isDragging
        if event == cv2.EVENT_MOUSEMOVE:
            if not isDragging:
                img_draw = img.copy()
                cv2.rectangle(img_draw, (x-half, y-half),
                              (x+half, y+half), (0, 255, 0))
                cv2.imshow(win_title, img_draw)
        elif event == cv2.EVENT_LBUTTONDOWN:
            isDragging = True
            cx1, cy1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            if isDragging:
                isDragging = False
                liquify(img, cx1, cy1, x, y)
                cv2.imshow(win_title, img)
                cv2.imwrite('C:/Temp/save_image.jpg', img)

    global img
    img = cv2.imread(get_path.get())
    h, w = img.shape[:2]

    cv2.namedWindow(win_title)
    cv2.setMouseCallback(win_title, onMouse)
    cv2.imshow(win_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#왜곡 거울 카메라 함수
def distotion_cam():
    msgbox.showinfo("편집방법", "스크린샷: Enter, 카메라 종료: Esc")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    rows, cols = 240, 320
    map_y, map_x = np.indices((rows, cols), dtype=np.float32)

    #거울 왜곡 효과
    map_mirrorh_x, map_mirrorh_y = map_x.copy(), map_y.copy()
    map_mirrorv_x, map_mirrorv_y = map_x.copy(), map_y.copy()
    ##좌우 대칭 거울 좌표 연산
    map_mirrorh_x[:, cols//2:] = cols - map_mirrorh_x[:, cols//2:]-1
    ##상하 대칭 거울 좌표 연산
    map_mirrorv_y[rows//2:, :] = rows - map_mirrorv_y[rows//2:, :]-1

    #물결 효과
    map_wave_x, map_wave_y = map_x.copy(), map_y.copy()
    map_wave_x = map_wave_x + 15 * np.sin(map_y/20)
    map_wave_y = map_wave_y + 15 * np.sin(map_x/20)

    #렌즈 효과
    ##렌즈 효과, 중심점 이동
    map_lenz_x = 2*map_x/(cols - 1) - 1
    map_lenz_y = 2*map_y/(rows - 1) - 1
    ##렌즈 효과, 극좌표 변환
    r, theta = cv2.cartToPolar(map_lenz_x, map_lenz_y)
    r_convex = r.copy()
    r_concave = r
    ##볼록 렌즈 효과 매핑 좌표 연산
    r_convex[r<1] = r_concave[r<1] ** 2
    print(r.shape, r_convex[r<1].shape)
    ##오목렌즈 효과 매핑 좌표 연산
    r_concave[r<1] = r_concave[r<1] ** 0.5
    ##렌즈 효과, 직교 좌표 복원
    map_convex_x, map_convex_y = cv2.polarToCart(r_convex, theta)
    map_concave_x, map_concave_y = cv2.polarToCart(r_concave, theta)
    ##렌즈 효과, 좌상단 좌표 복원
    map_convex_x = ((map_convex_x + 1) * cols - 1)/2
    map_convex_y = ((map_convex_y + 1) * rows - 1)/2
    map_concave_x = ((map_concave_x +1) * cols - 1)/2
    map_concave_y = ((map_concave_y + 1) * rows - 1)/2

    while True:
        ret, frame = cap.read()
        #준비한 매핑 좌표로 영상 효과 적용
        mirrorh = cv2.remap(frame, map_mirrorh_x,map_mirrorh_y,cv2.INTER_LINEAR)
        mirrorv = cv2.remap(frame, map_mirrorv_x,map_mirrorv_y, cv2.INTER_LINEAR)
        wave = cv2.remap(frame, map_wave_x,map_wave_y,cv2.INTER_LINEAR,
                         None, cv2.BORDER_REPLICATE)
        convex = cv2.remap(frame, map_convex_x, map_convex_y, cv2.INTER_LINEAR)
        concave = cv2.remap(frame, map_concave_x, map_concave_y, cv2.INTER_LINEAR)
        #영상 합치기
        r1 = np.hstack((frame, mirrorh, mirrorv))
        r2 = np.hstack((wave, convex, concave))
        merged = np.vstack((r1, r2))

        cv2.imshow('distorted', merged)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        elif cv2.waitKey(1) & 0xFF == 13:
            cv2.imwrite('C:/Temp/save_image.jpg', merged)
    cap.release
    cv2.destroyAllWindows()

#스케치 카메라 함수
def sketch_cam():
    msgbox.showinfo("편집방법", "스크린샷 전체:1, 흑백:2, 컬러:3, 카메라 종료: Esc")
    # 카메라 장치 연결
    cap = cv2.VideoCapture(0)   
    while cap.isOpened():
        # 프레임 읽기
        ret, frame = cap.read()
        # 속도 향상을 위해 영상크기를 절반으로 축소
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5,                             interpolation=cv2.INTER_AREA)
        if cv2.waitKey(1) == 27: # esc키로 종료
            break
        elif cv2.waitKey(1) & 0xFF == 49:
            cv2.imwrite('C:/Temp/save_image.jpg', merged)    
        elif cv2.waitKey(1) & 0xFF == 50:
            cv2.imwrite('C:/Temp/save_image.jpg', img_sketch)
        elif cv2.waitKey(1) & 0xFF == 51:
            cv2.imwrite('C:/Temp/save_image.jpg', img_paint)
        # 그레이 스케일로 변경    
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 잡음 제거를 위해 가우시안 플러 필터 적용(라플라시안 필터 적용 전에 필수)
        img_gray = cv2.GaussianBlur(img_gray, (9,9), 0)
        # 라플라시안 필터로 엣지 검출
        edges = cv2.Laplacian(img_gray, -1, None, 5)
        # 스레시홀드로 경계 값 만 남기고 제거하면서 화면 반전(흰 바탕 검은 선)
        ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

        # 경계선 강조를 위해 침식 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        sketch = cv2.erode(sketch, kernel)
        # 경계선 자연스럽게 하기 위해 미디언 블러 필터 적용
        sketch = cv2.medianBlur(sketch, 5)
        # 그레이 스케일에서 BGR 컬러 스케일로 변경
        img_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        # 컬러 이미지 선명선을 없애기 위해 평균 블러 필터 적용
        img_paint = cv2.blur(frame, (10,10) )
        # 컬러 영상과 스케치 영상과 합성
        img_paint = cv2.bitwise_and(img_paint, img_paint, mask=sketch)

        # 결과 출력
        merged = np.hstack((img_sketch, img_paint))
        cv2.imshow('Sketch Camera', merged)

    cap.release()
    cv2.destroyAllWindows()

#배경 분리 함수
def bg_seperate():
    msgbox.showinfo("편집방법", "자동 저장\n좌클릭+드래그: 일괄 제거\nSHIFT+드래그: 정밀제거\nCTRL+드래그: 제거부분정밀복구\nEsc: 종료") 
    img = cv2.imread(get_path.get())
    img_draw = img.copy()
    global mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # 마스크 생성
    global rect
    rect = [0,0,0,0]    # 사각형 영역 좌표 초기화
    mode = cv2.GC_EVAL  # 그랩컷 초기 모드
    # 배경 및 전경 모델 버퍼
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)

    # 마우스 이벤트 처리 함수
    def onMouse(event, x, y, flags, param):
        global mouse_mode, rect, mask, mode
        if event == cv2.EVENT_LBUTTONDOWN : # 왼쪽 마우스 누름
            if flags <= 1: # 아무 키도 안 눌렀으면
                mode = cv2.GC_INIT_WITH_RECT # 드래그 시작, 사각형 모드
                rect[:2] = x, y # 시작 좌표 저장
        # 마우스가 움직이고 왼쪽 버튼이 눌러진 상태
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON :
            if mode == cv2.GC_INIT_WITH_RECT: # 드래그 진행 중
                img_temp = img.copy()
                # 드래그 사각형 화면에 표시
                cv2.rectangle(img_temp, (rect[0], rect[1]), (x, y), (0,255,0), 2)
                cv2.imshow('img', img_temp)
            elif flags > 1: # 키가 눌러진 상태
                mode = cv2.GC_INIT_WITH_MASK    # 마스크 모드
                if flags & cv2.EVENT_FLAG_CTRLKEY :# 컨트롤 키, 분명한 전경
                    # 흰색 점 화면에 표시
                    cv2.circle(img_draw,(x,y),3, (255,255,255),-1)
                    # 마스크에 GC_FGD로 채우기
                    cv2.circle(mask,(x,y),3, cv2.GC_FGD,-1)
                if flags & cv2.EVENT_FLAG_SHIFTKEY : # 쉬프트키, 분명한 배경
                    # 검정색 점 화면에 표시
                    cv2.circle(img_draw,(x,y),3, (0,0,0),-1)
                    # 마스크에 GC_BGD로 채우기
                    cv2.circle(mask,(x,y),3, cv2.GC_BGD,-1)
                cv2.imshow('img', img_draw) # 그려진 모습 화면에 출력
        elif event == cv2.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼 뗀 상태
            if mode == cv2.GC_INIT_WITH_RECT : # 사각형 그리기 종료
                rect[2:] =x, y # 사각형 마지막 좌표 수집
                # 사각형 그려서 화면에 출력
                cv2.rectangle(img_draw, (rect[0], rect[1]), (x, y), (255,0,0), 2)
                cv2.imshow('img', img_draw)
            # 그랩컷 적용 ---⑧
            cv2.grabCut(img, mask, tuple(rect), bgdmodel, fgdmodel, 1, mode)
            img2 = img.copy()
            # 마스크에 확실한 배경, 아마도 배경으로 표시된 영역을 0으로 채우기
            img2[(mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD)] = 2
            cv2.imshow('grabcut', img2) # 최종 결과 출력
            cv2.imwrite('C:/Temp/save_image.jpg', img2)  
            mode = cv2.GC_EVAL # 그랩컷 모드 리셋
    # 초기 화면 출력 및 마우스 이벤트 등록
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', onMouse)
    while True:    
        if cv2.waitKey(0) & 0xFF == 27 : # esc
            break
    cv2.destroyAllWindows()

#포스터 제작 함수
def poster():
    msgbox.showinfo("편집방법", "자동 저장\nEsc: 종료") 
    img = cv2.imread(get_path.get())
    def onChange(x):
        sp = cv2.getTrackbarPos('sp', 'img')
        sr = cv2.getTrackbarPos('sr', 'img')
        lv = cv2.getTrackbarPos('lv', 'img')
        
        mean = cv2.pyrMeanShiftFiltering(img, sp, sr, None, lv)
        cv2.imshow('img', mean)
        cv2.imwrite('C:/Temp/save_image.jpg', mean)  
        
    cv2.imshow('img', img)
    cv2.createTrackbar('sp', 'img', 0, 100, onChange)
    cv2.createTrackbar('sr', 'img', 0, 100, onChange)
    cv2.createTrackbar('lv', 'img', 0, 5, onChange)
    while True:    
        if cv2.waitKey(0) & 0xFF == 27 : # esc
            break
    cv2.destroyAllWindows()

#파노라마 왼쪽 사진 선택 함수
def panorama_Left():
    window.file = filedialog.askopenfile(
    initialdir='path',
        title='select file',
        filetypes=(('all files', '*.*'), ('png files', '*.png'), ('jpg files','*.jpg')))
    print(window.file.name)
    
    panoramaLeftEnt.delete(0, END) 
    #해당 코드가 없으면 이미지를 불러올 때마다 경로 입력창 뒤에 계속해서 새로 선택한 이미지 경로가 붙는다.
    
    panoramaLeftEnt.insert(0, window.file.name)

#파노라마 오른쪽 사진 선택 함수
def panorama_Right():
    window.file = filedialog.askopenfile(
    initialdir='path',
        title='select file',
        filetypes=(('all files', '*.*'), ('png files', '*.png'), ('jpg files','*.jpg')))
    print(window.file.name)
    
    panoramaRightEnt.delete(0, END) 
    #해당 코드가 없으면 이미지를 불러올 때마다 경로 입력창 뒤에 계속해서 새로 선택한 이미지 경로가 붙는다.
    
    panoramaRightEnt.insert(0, window.file.name)    

#파노라마 실행 함수
def panorama():
    # 왼쪽/오른쪽 사진 읽기
    # 왼쪽 사진 = train = 매칭의 대상
    imgL = cv2.imread(panoramaLeftEnt.get())
    # 오른쪽 사진 = query = 매칭의 기준
    imgR = cv2.imread(panoramaRightEnt.get())

    hl, wl = imgL.shape[:2]     # 왼쪽 사진 높이, 넓이
    hr, wr = imgR.shape[:2]     # 오른쪽 사진 높이, 넓이

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)  # 그레이 스케일 변환
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)  # 그레이 스케일 변환

    # SIFT 특징 검출기 생성 및 특징점 검출
    descriptor = cv2.xfeatures2d.SIFT_create()  # SIFT 추출기 생성
    (kpsL, featuresL) = descriptor.detectAndCompute(imgL, None) # 키포인트, 디스크립터 
    (kpsR, featuresR) = descriptor.detectAndCompute(imgR, None) # 키포인트, 디스크립터 

    # BF 매칭기 생성 및 knn 매칭
    matcher = cv2.DescriptorMatcher_create("BruteForce")    # BF 매칭기 생성
    matches = matcher.knnMatch(featuresR, featuresL, 2)     # knn 매칭

    # 좋은 매칭점 선별
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:   ### 75%
            good_matches.append((m[0].trainIdx, m[0].queryIdx))

    print('matches:{}/{}'.format(len(good_matches), len(matches)))

    # 좋은 매칭점이 4개 이상인 원근 변환행렬 구하기
    if len(good_matches) > 4:
        ptsL = np.float32([kpsL[i].pt for (i,_) in good_matches])
        ptsR = np.float32([kpsR[i].pt for (_,i) in good_matches])
        mtrx, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)

        # 원근 변환행렬로 오른쪽 사진을 원근 변환, 결과 이미지 크기는 사진 2장 크기
        panorama = cv2.warpPerspective(imgR, mtrx, (wr + wl, hr))

        # 왼쪽 사진을 원근 변환한 왼쪽 영역에 합성
        panorama[0:hl, 0:wl] = imgL
    else:
        panorama = imgL

    # 결과 출력
    cv2.imshow('origin Left', imgL)
    cv2.moveWindow('origin Left', 0, 0)
    
    cv2.imshow('origin Right', imgR)
    cv2.moveWindow('origin Right', 650, 0)
    
    cv2.imshow('Panorama', panorama)
    cv2.moveWindow('Panorama', 0, 530)
    cv2.imwrite('C:/Temp/save_image.jpg', panorama)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
filem = tk.Menu(mainmenu, tearoff=0)
filem.add_command(label='종료', command = close_window)
mainmenu.add_cascade(label='실행', menu=filem)
window.config(menu = mainmenu)

image_path = Label(window, text="이미지 선택:")
image_path.place(x=10, y=10)

get_path = Entry(window)
get_path.place(x=90, y=10)

path_button = Button(window, text="이미지 선택", command=file_choose)
path_button.place(x=240, y=10)

img_preview = Label(window, text="편집할 이미지 미리보기")
img_preview.place(x=10, y=40)

photo = Image.open("C:/Temp/previewImage.png")
resize_photo = photo.resize((300, 300))#미리보기 이미지 사이즈 조정
final_photo = ImageTk.PhotoImage(resize_photo, master = window)
image_label = Label(window, image=final_photo)
image_label.place(x=10, y=70)   

test_button = Button(window, text="편집 이미지 확인", command=edit_image)
test_button.place(x=360, y=10)

photo2 = Image.open("C:/Temp/previewImage.png")
resize_photo2 = photo2.resize((300, 300))#미리보기 이미지 사이즈 조정
final_photo2 = ImageTk.PhotoImage(resize_photo2, master = window)
editimg_label = Label(window, image=final_photo2)
editimg_label.place(x=360, y=70)  

#크로마키 합성 GUI
editimg1 = Label(window, text="크로마키 합성:")
editimg1.place(x=10, y=400)

editimg1Ent = Entry(window)
editimg1Ent.place(x=100, y=400)

chromaselBtn = Button(window, text="크로마키 선택", command=chroma_select)
chromaselBtn.place(x=250, y=385)

editimg1Btn = Button(window, text="실행", command=chromakey)
editimg1Btn.place(x=250, y=410)

#스티커 합성 GUI
editimg2 = Label(window, text="스티커 합성:")
editimg2.place(x=10, y=465)

editimg2Ent = Entry(window)
editimg2Ent.place(x=100, y=465)

stickerselBtn = Button(window, text="스티커 선택", command=sticker_select)
stickerselBtn.place(x=250, y=450)

editimg2Btn = Button(window, text="실행", command=SeamlessClone)
editimg2Btn.place(x=250, y=475)

#모자이크 합성 GUI
editimg3 = Label(window, text="모자이크 합성:")
editimg3.place(x=10, y=530)

editimg3Btn = Button(window, text="실행",  command=mosaic)
editimg3Btn.place(x=100, y=528)

#리퀴파이 합성 GUI
editimg4 = Label(window, text="리퀴파이 합성:")
editimg4.place(x=10, y=595)

editimg4Btn = Button(window, text="실행", command=liquifys)
editimg4Btn.place(x=100, y=593)

#왜곡 거울 카메라 GUI
editimg5 = Label(window, text="왜곡 거울 카메라:")
editimg5.place(x=10, y=660)

editimg5Btn = Button(window, text="실행", command=distotion_cam)
editimg5Btn.place(x=115, y=658)

#스케치 효과 카메라 GUI
editimg6 = Label(window, text="스케치 카메라:")
editimg6.place(x=360, y=400)

editimg6Btn = Button(window, text="실행", command=sketch_cam)
editimg6Btn.place(x=450, y=398)

#배경 분리 GUI
editimg7 = Label(window, text="배경화면 분리:")
editimg7.place(x=360, y=465)

editimg7Btn = Button(window, text="실행", command=bg_seperate)
editimg7Btn.place(x=450, y=463)

#인스턴트 포스터 제작 GUI
editimg8 = Label(window, text="포스터 제작:")
editimg8.place(x=360, y=530)

editimg8Btn = Button(window, text="실행", command=poster)
editimg8Btn.place(x=438, y=528)

insert_figlabel = Label(window, text="편집 이미지 도형 삽입:")
insert_figlabel.place(x=360, y=660)
figure_button = Button(window, text="실행", command=insert_figure)
figure_button.place(x=495, y=658)

#파노라마 이미지 제작 GUI
panoramaLabel = Label(window, text="파노라마 이미지:")
panoramaLabel.place(x=360, y=595)

editimg9Btn = Button(window, text="실행", command=panorama)
editimg9Btn.place(x=460, y=593)

panoramaLeft = Label(window, text="왼쪽 이미지:")
panoramaLeft.place(x=370, y=615)

panoramaLeftEnt = Entry(window)
panoramaLeftEnt.place(x=460, y=615)

panoramaLeftBtn = Button(window, text="선택", command=panorama_Left)
panoramaLeftBtn.place(x=610, y=605)

panoramaRight = Label(window, text="오른쪽 이미지:")
panoramaRight.place(x=370, y=635)

panoramaRightEnt = Entry(window)
panoramaRightEnt.place(x=460, y=635)

panoramaRightBtn = Button(window, text="선택", command=panorama_Right)
panoramaRightBtn.place(x=610, y=635)

window.mainloop()


# In[ ]:




