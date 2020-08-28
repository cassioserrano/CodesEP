import cv2      # Importa o OpenCV

print('------------------------------------------------------------------------------\n')                      # 
print('     Bem vindo ao programa de Reconhecimento facial com Python + OpenCV !!!      \n')                   # "Interface" inicial
print(' | Você pode escolher uma imagem da sua GALERIA ou fazer o reconhecimento em tempo real pela WEBCAN ')  # 
escolha = str(input(' | Digite GALERIA ou WEBCAN para escolher: ').lower())   # Escolha do usuário

frontalface_path = 'haarcascade_frontalface_default.xml'   # Arquivo .xml que faz a detecção facial
eyes_path = 'haarcascade_eye.xml'   # Arquivo .xml que faz a detecção dos olhos

face_cascata = cv2.CascadeClassifier(frontalface_path)
olhos_cascata = cv2.CascadeClassifier(eyes_path)

#-- O usuário escolheu GALERIA --#
    
if escolha == 'galeria':     
        print(' | Escolha uma imagem da sua galeria')          

        try:
            foto = input(' |    - Nome da imagem completa com a extensão (Exemplo: foto.jpg): ').lower()  # Escolha de uma foto na mesma pasta que o programa esta armazenado
           
            image_path = foto   # Variável que armazena a foto escolhida

            img = cv2.imread(image_path)    # Lê a imagem escolhida pixel a pixel

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Transforma a imagem em escala cinza para melhorar o processo de identificação
            gray = cv2.equalizeHist(gray)
            faces = face_cascata.detectMultiScale(gray, 1.25, 10)

            for(x, y, w, h) in faces:   # Cria o retangulo para demarcar a face
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)  # Insere as coordenadas da face
            
                roi = gray[y:y+h,x:x+w]
                eyes = olhos_cascata.detectMultiScale(roi)
                
                for(x2, y2, w2, h2) in eyes:
                    centro_olhos = (x + x2 + w2//2, y + y2 + h2//2)
                    radianos = int(round((w2 + h2)*0.25))   # Tamanho do circulo que identifica os olhos
                    img = cv2.circle(img, centro_olhos, radianos, (255, 0, 0), 2)   # Insere as coordenadas dos olhos
            
            print(' | Aperte a tecla "q" para sair')   # Finalizar o programa

            cv2.imshow('imagem', img)   # Mostra a imagem na tela com o retangulo
            if cv2.waitKey(0) & 0xFF == ord('q'):   #-- Fecha o Programa
                print(' | Até a proxima ;) ')
                cv2.destroyAllWindows()                 #
            
        except: # Tratamento de erro quando o nome da imagem esta errado
           print(' | *Oops! Nome da imagem incorreta, verifique e tente novamente...')

#-- Usuário escolheu WEBCAN --#
    
elif escolha == 'webcan':       

        print(' | SORRIA... X')

        webcan = cv2.VideoCapture(0)    # Abre a webcan

        print(' | Aperte a tecla "q" para sair')

        while True:

            ret, img = webcan.read()  # Leitura dos pixels da imagem gerada pela webcan
            img = cv2.flip(img, 180)    # Inverte a imagem gerada pela webcan em 180 para ficam parecido com o reflexo de um espelho

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Transforma a imagem em escala cinza para melhorar o processo de identificação
            gray = cv2.equalizeHist(gray)
            faces = face_cascata.detectMultiScale(gray, minNeighbors = 6, minSize = (30, 30), maxSize = (200, 200))

            for(x, y, w, h) in faces:   # Cria o retangulo para demarcar a face
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

                roi = gray[y:y+h,x:x+w]
                eyes = olhos_cascata.detectMultiScale(roi)  
                for(x2, y2, w2, h2) in eyes:
                    centro_olhos = (x + x2 + w2//2, y + y2 + h2//2)
                    radianos = int(round((w2 + h2)*0.25))
                    img = cv2.circle(img, centro_olhos, radianos, (255, 0, 0), 2)

        
            cv2.imshow('image', img)    # Mostra a imagem em tempo real        
            if cv2.waitKey(1) & 0xFF == ord('q'):   # Fecha o programa
                print(' | Até a proxima ;) ')       #
                break                               #
        webcan.release()    # Fecha a webcan
        cv2.destroyAllWindows()

#-- Tratamento caso o usuário nao escolher entre GALERIA ou WEBCAN --#
else:   
    print(' | *Oops! Expressão errada, execute novamente... ')   