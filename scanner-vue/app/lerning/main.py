from robot.Despachante import ScannerDespachante, LerningDigits
import os

paths = [os.path.join("img03", name) for name in os.listdir("img03")]

scanner = ""

for index, path in enumerate(paths, start=0):
    if index > 2:
        scanner = ScannerDespachante(str(path))
        scanner.forceHSV()
        scanner.morphologyEx()
