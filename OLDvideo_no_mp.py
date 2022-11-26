import cv2
import numpy as np

# blur = 21
# canny_low = 15
# canny_high = 150
# min_area = 0.0005
# max_area = 0.95
# dilate_iter = 10
# erode_iter = 10
# mask_color = (0.0,0.0,0.0)

def video_bg():
    #video temporario ate consertar captura da webcam no WSL -> perguntar se pode usar video gravado previo ao inves de ao vivo
    vcap = cv2.VideoCapture('videos/video2.mp4')

    while True:
        #success = booleano se o video normal funcionou
        #frame = frame do video
        success, frame = vcap.read()
        if success == True:
            img_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(img_gs, 5, 50)
            edge = cv2.dilate(edge, None)
            edge = cv2.erode(edge, None)
            #pega todos (chain_approx_none) os contornos ([0]) sem classificação hierárquica (retr_list)
            for c in cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
                info_c = [(c, cv2.contourArea(c))]
            img_ar = frame.shape[0] * frame.shape[1]
            print(img_ar)
            max_ar = 0.95 * img_ar
            min_ar = 0.005 * img_ar

            mask = np.zeros(edge.shape, dtype=np.uint8)
            for cont in info_c:
                if cont[1] > min_ar and cont[1] < max_ar:
                    #preenche a mascara com os contornos
                    mask = cv2.fillConvexPoly(mask, cont[0], (255))
            # mask = cv2.dilate(mask, None, iterations=5)
            # mask = cv2.erode(mask, None, iterations=5)
            # mask = cv2.GaussianBlur(mask, (21,21), 0)

            stack = np.dstack([mask]*3)
            stack = stack.astype('float32') /255.0
            frame = frame.astype('float32') /255.0
            print(stack.shape)
            
            video_mask = (stack * frame) + ((1-stack) * (0.0,0.0,0.0))
            video_mask = (video_mask * 255).astype('uint8')

            cv2.imshow("Frente", video_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
        else:
            break
        
video_bg()