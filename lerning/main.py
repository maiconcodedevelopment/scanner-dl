from robot.Despachante import ScannerDespachante , LerningDigits

# scanner = ScannerDespachante("img05/Scanner_20180905_27.png")
# scanner.forceHSV()
# scanner.morphologyEx()
# scanner.linhaRemove()
# scanner.images_read(path="numbers",list_image=["Layer 1.jpg","Layer 2.jpg","Layer 3.jpg","Layer 4.jpg","Layer 6.jpg","Layer 7.jpg","Layer 8.jpg","Layer 9.jpg","Layer 10.jpg","Layer 11.jpg"])
# scanner.inRange([0,0,0],[140,140,140],mask=True)
# scanner.morphologyEx()
# scanner.showImage()

scanner1 = ScannerDespachante("images/Scanner_20180820_2.png")
scanner1.showImage()
scanner1.forceHSV()
scanner1.morphologyEx()
# scanner1.inRange([0,0,0],[120,120,120],mask=True)
# scanner1.showImage()


# scanner1 = ScannerDespachante("images/Scanner_20180820 (5)-1.png")
# scanner1.forceHSV()
# scanner1.morphologyEx()


# lerning = LerningDigits()