build:
	bundle exec jekyll build
push:
	cd _site && \
	git add . && \
	git commit --message="Publish @$$(date)" &&\
	git push -u origin master
