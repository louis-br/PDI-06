import cv2

def main():
    file = 'rodovia2'
    #abrir o arquivo
    vcap = cv2.VideoCapture('videos/' + file +'.mp4')
    #bool se arquivo abriu e frame atual
    ret, frame = vcap.read()
    prev_frame = frame

    frame_diff = cv2.absdiff(frame,prev_frame)

    while True:
        if not ret:
            break
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(curr_frame_gray,prev_frame_gray)

        cv2.imshow('frame diff',frame_diff)
        cv2.imshow('og', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame.copy()
        ret, frame = vcap.read()

    cv2.imwrite("bg/fd-frame-" +file + ".png", frame)
    cv2.imwrite("bg/framediff-" +file +".png", frame_diff)


main()