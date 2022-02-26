#!/usr/bin/python

import sys, re

def edit(folder_name):
# 	path = "D:/GitHub/robertpiazza.github.io/_posts/" + str(sys.argv[1])
    path = "D:/GitHub/robertpiazza.github.io/_posts/2022-02-25-Deep Learning Part 4.md"
    yaml = "---\nlayout: post\ntitle: TITLE\ndate: YYYY-MM-DD HH:MM:SS -0700\ncategories:\n---\n\n"
    with open(path, 'r') as file:
        filedata = file.read()
    filedata = re.sub(r"!\[png\]\(", "<img src=\"/assets/images/"+folder_name+"/", filedata)
    filedata = re.sub(".png\)", ".png\">", filedata)
    filedata = yaml + filedata
    with open(path, 'w') as file:
        file.write(filedata)

if __name__ == '__main__':
    # folder_name = input("String with this post's folder name?")
    folder_name = "DLwPCh4"
    edit(folder_name)
