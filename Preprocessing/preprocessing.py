import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 불러오기
src = cv2.imread("./resource/easy0.png")

# 그레이스케일 이미지 생성, 이진화, 가로 세로 정의
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
height, width = gray.shape

# 마스크 정의와 레이블링
mask = np.zeros(gray.shape, np.uint8)
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)

# 보표 이미지 저장
# for i in range(1, cnt):
#     x, y, w, h, area = stats[i]
#     if w > width * 0.5:
#         roi = src[y-5:y+h+5, x-50:x+w+50]
#         cv2.imwrite('save/%s.PNG' %i, roi)

for i in range(1, cnt):
    x, y, w, h, area = stats[i]
    if w > width * 0.5:
        cv2.rectangle(mask, (x, y, w, h), (255, 255, 255), -1)

# 보표 추출
masked = cv2.bitwise_and(gray, mask)

# 오선 제거(히스토그램 방식)
staves = []
for row in range(height):
    pixels = 0
    for col in range(width):
        pixels += (masked[row][col] == 255)
    if pixels >= width * 0.5:
        if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:
            staves.append([row, 0])
        else:
            staves[-1][1] += 1

for staff in range(len(staves)):
    top_pixel = staves[staff][0]
    bot_pixel = staves[staff][0] + staves[staff][1]
    for col in range(width):
        if masked[top_pixel - 1][col] == 0 and masked[bot_pixel + 1][col] == 0:
            for row in range(top_pixel, bot_pixel + 1):
                masked[row][col] = 0

sta = []
for i in range(1, cnt):
    x, y, w, h, area = stats[i]
    if w > width * 0.5:
        y2= y+(h//2)
        roi1 = masked[y:y2, x:x+w]
        roi2 = masked[y2:y+h, x:x+w]
        sta.append(roi1)
        sta.append(roi2)
        # 객체 저장
        cv2.imwrite('save/1_%s.PNG' %i, 255-roi1)
        print('save/1_%s.PNG saved' %i)
        cv2.imwrite('save/2_%s.PNG' %i, 255-roi2)
        print('save/2_%s.PNG saved' %i)

# 보표 그래프 저장
j=1
histograms = []
for st in sta:
    height, width = st.shape
    histogram = np.zeros(width, np.uint8)

    for col in range(width-1):
        for row in range(height):
            if st[row, col] != 0:
                histogram[col] += 1
    histograms.append(histogram)
    plt.figure(figsize=(10, 5))
    plt.plot(histogram)
    # plt.show()
    plt.savefig('save/graph%s.PNG' %j)
    print('save/graph%s.PNG saved' %j)
    j+=1

j=1
for img in sta:    
    height, width = img.shape

    # 구간 찾기
    cut_points = []
    start = None
    for x in range(width):
        if not img[:, x].any():
            if start is None:
                start = x
        else:
            if start is not None:
                if x - start >= int(width / 500):
                    cut_points.append((start, x))
                start = None

    if start is not None and x - start >= int(width / 500):
        cut_points.append((start, x))

    # 이미지 자르기
    cut_imgs = []
    last_end = 0
    for start, end in cut_points:
        cut_imgs.append(img[:, last_end:start])
        last_end = end

    if last_end < width:
        cut_imgs.append(img[:, last_end:])

    # 이미지 저장
    for i, cut_img in enumerate(cut_imgs):
        bordered_img = cv2.copyMakeBorder(cut_img, 20, 20, 50, 50, cv2.BORDER_CONSTANT, None, value=0)
        filename = f'save/{j}bordered_img_{i}.jpg'
        cv2.imwrite(filename, 255-bordered_img)
        print('save/%s.jpg saved' %filename)
    j+=1