import os

with open("images.txt", "w") as a:
    for path, subdirs, files in os.walk(r'/home/henning/Desktop/images/val'):
        for filename in files:
            a.write(str(filename) + os.linesep)