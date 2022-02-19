#!/usr/bin/python

import sys, re

def edit(folder_name):
	path = "/Users/rober/Documents/GitHub/robertpiazza.github.io/_jupyter/" + str(sys.argv[1])
	yaml = "---\nlayout: post\ntitle: TITLE\ndate: YYYY-MM-DD HH:MM:SS -0700\nmathjax: true\ncategories:\n  - category\ntags:\n  - tag\n---\n\n"
	with open(path, 'r') as file:
		filedata = file.read()
	filedata = re.sub(r"!\[png\]\(", "<img src=\"/assets/images/"+folder_name+"/", filedata)
	filedata = re.sub(".png\)", ".png\">", filedata)
	filedata = yaml + filedata
	with open(path, 'w') as file:
		file.write(filedata)

if __name__ == '__main__':
    folder_name = input("String with this post's folder name?")
    edit(folder_name)
