import json
import cv2
import numpy as np

class Rectangle:
    def __init__(self, y=0, x=0, y2=None, x2=None, height=0, width=0):
        self.y = y
        self.x = x
        self.y2 = y2 if y2 else y + height
        self.x2 = x2 if x2 else x + width

    def from_dict(self, dict):
        self.__init__(y=dict['y'], x=dict['x'], height=dict['height'], width=dict['width'])
    
    def height(self):
        return self.y2 - self.y

    def width(self):
        return self.x2 - self.x

    def area(self):
        return self.height()*self.width()

    def mid_y(self):
        return self.y + self.height()//2

    def intersection(self, other):
        a = self
        b = other
        #ax1, ay1, ax2, ay2 = self.x, self.y, self.x2, self.y2
        #bx1, by1, bx2, by2 = other.x, other.y, other.x2, other.y2
        x1 = max(min(a.x, a.x2), min(b.x, b.x2))
        y1 = max(min(a.y, a.y2), min(b.y, b.y2))
        x2 = min(max(a.x, a.x2), max(b.x, b.x2))
        y2 = min(max(a.y, a.y2), max(b.y, b.y2))
        return Rectangle(y=y1, x=x1, y2=y2, x2=x2) if x1 < x2 and y1 < y2 else None


def get_roi_from_json(path):
    with open(path, 'r') as file:
        rect = Rectangle()
        rect.from_dict(json.load(file))
        return rect


class LineCarCounter:
    def __init__(self, ROI_path, min_height=50, min_width=50): #, line_size=25, line_y_percent=0.5):
        self.ROI = get_roi_from_json(ROI_path)
        self.min_height = min_height
        self.min_width = min_width
        self.last_contour_count = 0
        self.cars = 0
        #self.ROI.y = int((self.ROI.y + self.ROI.height) * line_y_percent - line_size/2)
        #self.ROI.height = line_size//2

    def update(self, frame):
        frame = frame[self.ROI.y:self.ROI.y2, self.ROI.x:self.ROI.x2]
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)*255, contours, -1, (255,0,0), 3)
        count = 0
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w < self.min_width and h < self.min_height:
                continue
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            count = count + 1
        
        new = max(self.last_contour_count - count, 0)
        self.last_contour_count = count
        self.cars += new

        cv2.imshow("contours", img)

        print(f'Cars: {self.cars} New: {self.last_contour_count - count}')
        return new


class TrackingCarCounter:
    def __init__(self, ROI_path, min_height=25, min_width=25):
        self.ROI = get_roi_from_json(ROI_path)
        self.ROI_mid = self.ROI.height()//2
        self.min_height = min_height
        self.min_width = min_width
        self.current_id = 0
        self.cars = 0
        self.last_trackers = []
        self.used_ids = {}

    def update(self, frame):
        frame = frame[self.ROI.y:self.ROI.y2, self.ROI.x:self.ROI.x2]
        # print(frame.type)
        #cv2.imshow("crop", frame*255)
        trackers = []
        new = 0
        found_contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.drawContours(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)*255, found_contours, -1, (255,0,0), 3)
        for contour in found_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w < self.min_width and h < self.min_height:
                continue
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            rect = Rectangle(y=y, x=x, height=h, width=w)
            tracker = {
                'contour': contour,
                'rect': rect,
                'mid_y': rect.mid_y(),
                'id': 0
            }
            trackers.append(tracker)
            best, best_area = None, 0
            for old_tracker in self.last_trackers:
                intersection = old_tracker['rect'].intersection(rect)
                area = intersection.area() if intersection else 0
                #print("Area: ", area)
                if area > best_area:
                    best = old_tracker
                    best_area = area
            
            #print()
            if not best:
                self.current_id += 1
                tracker['id'] = self.current_id
                continue

            tracker['id'] = best['id']

            line_crossed = np.sign(best['mid_y'] - self.ROI_mid) != np.sign(tracker['mid_y'] - self.ROI_mid)
            
            if line_crossed and tracker['id'] not in self.used_ids:
                self.used_ids[tracker['id']] = True
                new += 1

        self.last_trackers = trackers
        self.cars += new
        cv2.imshow("contours", img)
        print(f'Cars: {self.cars} New: {new}')
        return new