cd notebooks
for F in $(ls ./*.ipynb); do
    FBASE=$(basename $F)
    FNAME="${FBASE%.*}"

    # convert
    jupyter nbconvert --to markdown $F

    # fix kramdown bug and image paths
    sed -i 's/\\{/\\\\{/g' $FNAME.md
    sed -i 's/\\}/\\\\}/g' $FNAME.md
    sed -i "s/${FNAME}_files/\/images\/${FNAME}_files/g" $FNAME.md
    #sed -i "s/${FNAME}_files\///g" $FNAME.md

    # organize
    mv $FNAME.md ../_posts/
    [ -d "../images/${FNAME}_files" ] && rm -rf "../images/${FNAME}_files"
    mv "${FNAME}_files" ../images/
    cd "../images/${FNAME}_files/"
    optipng -o7 -strip all *.png
    #mv $FNAME.md "${FNAME}_files"/
    #mv "${FNAME}_files" $FNAME
    #[ -d "../_posts/$FNAME" ] && rm -rf "../_posts/$FNAME"
    #mv $FNAME ../_posts
done
cd ..
