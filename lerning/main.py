from robot.Despachante import ScannerDespachante

scanner = ScannerDespachante("images/Scanner_20180827_23.png")
scanner.forceHSV()
# scanner.linhaRemove()
# scanner.images_read(path="numbers",list_image=["Layer 1.jpg","Layer 2.jpg","Layer 3.jpg","Layer 4.jpg","Layer 6.jpg","Layer 7.jpg","Layer 8.jpg","Layer 9.jpg","Layer 10.jpg","Layer 11.jpg"])
# scanner.inRange([0,0,0],[140,140,140],mask=True)
# scanner.morphologyEx()
# scanner.showImage()

# scanner1 = ScannerDespachante("img03/Scanner_20180903_2.png")
# scanner1.inRange([0,0,0],[120,120,120],mask=True)
# scanner1.showImage()