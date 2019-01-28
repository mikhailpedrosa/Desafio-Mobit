import cv2

img = cv2.imread("objetos.png")
img_gauss = cv2.GaussianBlur(img, (5,5), 0)

img_gauss[img_gauss > 180] = 255
img_gauss[img_gauss <= 180] = 0

img_canny = cv2.Canny(img_gauss, 50, 160)

_, contornos, _ = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contornos, -1, (0, 0, 255), 2)
cv2.imshow("Foi encontrado {0} objetos na imagem.".format(len(contornos)), img)
cv2.imwrite("result_q1_quant_img_{0}.png".format(len(contornos)), img)
cv2.waitKey()