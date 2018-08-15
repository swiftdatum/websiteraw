build:
	bundle exec jekyll build

serve:
	bundle exec jekyll serve

push:
	cd _site && \
	git add . && \
	git commit --message="Publish @$$(date)" &&\
	git push -u origin master

notebooks:
	./notebookstoposts.sh

all: notebooks build push
	echo "All done!"
