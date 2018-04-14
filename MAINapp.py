#Initialiserer moduler
import numpy as np
import cv2
import Person as ps #Class Person og MultiPerson
import time

#Variabler til at håndtere mængden af passagerer
cnt_up = 0
cnt_down = 0

#Opretter forbindelse til videokilden
videokilde = cv2.VideoCapture(0)

#Beregner data om videokilden
w = videokilde.get(3)
h = videokilde.get(4)
frameArea = h*w
areaTH = frameArea/250 #Bestemmer passende størrelse for en person
print(areaTH, 'px')

#Linjer til bestemmelse af retning
line_up = int(2*(h/5))
line_down   = int(3*(h/5))

#Begrænsning for tracking af personernes retning
up_limit =   int(1*(h/5))
down_limit = int(4*(h/5))

#Variabler til grafiske elementer
line_down_color = (76,153,0)
line_up_color = (0,204,204)
pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

font = cv2.FONT_HERSHEY_SIMPLEX

#Initialiserer et baggrundssubtraktionsalgoritme
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#Variabler til håndtering af unikke personID'er
personer = []
max_p_age = 5
pid = 1 

#Den primære funktion - main() starter
def main():

    #Der gives adgang til dynamiske variabler uden for funktionen
    global cnt_up, cnt_down, pid
    
    while True:
        ret, frame = videokilde.read() #Returnerer det øjeblikkelige billede fra videokilden under variabelnavnet 'frame'

        #Ælder alle unikke personID'er med 1 arbitrær enhed
        for i in personer:
            i.age_one()

        #Pålægger baggrundssubtraktionsalgoritmet. Derudover itereres funktionen endnu en gang for at fjerne skygger og refleksioner
        fgmask = fgbg.apply(frame)
        fgmask2 = fgbg.apply(frame)
        
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)

        mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
        
        #Finder konturer for alle elementer tilbage efter MOG2-algoritmet
        _, contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours0:
            area = cv2.contourArea(cnt)
            if area > areaTH: #Filtrerer elementet fra, hvis det formentlig ikke er en person
                
                #Finder centrum af elementet
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x,y,w,h = cv2.boundingRect(cnt)

                #Tjekker om elementet er en ny person. Hvert personID er oprettet som et object af class'en Person fra Person.py
                new = True
                if cy in range(up_limit,down_limit):
                    for i in personer:
                        if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:

                            #For alle pre-registrerede personer, monitoreres retningen
                            new = False
                            i.updateCoords(cx,cy)
                            if i.going_UP(line_down,line_up) == True: #Betingelsen opfyldes hvis en person klassificieres som gående opad
                                cnt_up += 1
                                print("ID:",i.getId(),'Gik op ved tidspunktet',time.strftime("%c"))
                            elif i.going_DOWN(line_down,line_up) == True: #Betingelsen opfyldes hvis en person klassificeres som gående nedad
                                cnt_down += 1 
                                print("ID:",i.getId(),'Gik ned ved tidspunktet',time.strftime("%c"))
                            break
                        
                        #For alle elementer hvis retningen ikke er endegyldigt klassificeret endnu logges data for hvert personID
                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > down_limit:
                                i.setDone()
                            elif i.getDir() == 'up' and i.getY() < up_limit:
                                i.setDone()
                        
                        if i.timedOut(): #Rydder op i personID'erne                            
                            index = personer.index(i)
                            personer.pop(index)
                            del i 
                    
                    if new == True: #Alle resterende elementer, der ikke er set før, tilføjes som et nyt personID.
                        p = ps.Person(pid,cx,cy, max_p_age)
                        personer.append(p)
                        pid += 1     

                #Personspecifikke grafiske elementer
                cv2.circle(frame,(cx,cy), 5, (0,0,255), -1) #Tegner centrum af hver registreret person

                #img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) #Tegner det omsluttende rektangel for hver registreret person
                            
                cv2.drawContours(frame, cnt, -1, (0,255,0), 3) #Tegner konturen for hver registreret person 
                

        for i in personer:
            """
            #Tegner et spor efter hver person
            if len(i.getTracks()) >= 2:
            pts = np.array(i.getTracks(), np.int32)
            pts = pts.reshape((-1,1,2))
            frame = cv2.polylines(frame,[pts],False,i.getRGB())
            if i.getId() == 9:
                print(str(i.getX()), ',', str(i.getY()))
            """
            #Tegner hver persons associerede personID
            cv2.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)
            
        #Generelle grafiske elementer

        #Bestemmer tekstelementer
        str_up = 'Op: '+ str(cnt_up)
        str_down = 'Ned: '+ str(cnt_down)

        #Tegner streger, der indikerer tracking-begrænsningen og retningsbestemmelsen
        frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)

        #Op:/Ned: tekst med hvid outline
        cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str_up ,(10,40),font,0.5,line_up_color,1,cv2.LINE_AA)
        cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame, str_down ,(10,90),font,0.5,line_down_color,1,cv2.LINE_AA)

        #Viser det endelige billede
        cv2.imshow('Passagerregistrering',frame)

        #Illustrerer resultatet af baggrundssubtraktionsalgoritmet
        cv2.imshow('BGSubtraktion',mask)

        #Terminerer while-loopet
        if cv2.waitKey(25) & 0xff == ord('q'):
            break

    #Slukker forbindelsen til videokilden og lukker alle åbne vinduer
    videokilde.release()
    cv2.destroyAllWindows()

#main() slutter



#Tjekker om scriptet kun bruges til initialisering af variabler
if __name__ == '__main__':
    main()