build:
	bundle exec jekyll build
push:
	cd _site && \
	git add . && \
	git commit --message="Publish @$$(date)" &&\
	git push -u origin master

notebooks:
	bash notebookstoposts.sh

all: build notebooks push
	echo "All done!"
